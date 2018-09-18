import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

#np.random.seed(1)
#torch.manual_seed(1)


# mask is an index matrix that refers to action
def build_mask(n_actions, index):
    batch_size = index.size()[0]
    index = index.view(1, -1)
    index = index.to(torch.uint8)
    mask = torch.zeros(batch_size, n_actions)
    for i in range(batch_size):
        mask[i, index[0, i].numpy()] = 1

    return mask.to(torch.uint8)
    

# Build neural network: The structure can be modified in this class
# 'n_actions' is the output action vector
# 'n_features' is the input state vector
class NN_model(nn.Module):

    def __init__(self, n_features, n_actions, dueling):
        super(NN_model, self).__init__()
        self.dueling = dueling
        self.forward1 = nn.Linear(n_features, 20)
        self.Relu = nn.ReLU()
        # weather to use dueling DQN
        if self.dueling:
            self.Value = nn.Linear(20, 1)
            self.Advantage = nn.Linear(20, n_actions)
        else:
            self.forward2 = nn.Linear(20, n_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.3)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.forward1(x)
        x = self.Relu(x)
        if self.dueling:
            x = self.Value(x) + (self.Advantage(x) - torch.mean(self.Advantage(x), 1, keepdim=True))
        else:
            x = self.forward2(x)

        return x


class SumTree(object):

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
    
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree

    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
    
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            use_pretrained=False,
            use_double_q=False,
            prioritized=False,
            dueling=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.8 if e_greedy_increment is not None else self.epsilon_max
        self.path_to_para = 'DQN_para'
        self.use_double_q = use_double_q
        self.prioritized = prioritized
        self.dueling = dueling

        # total learning step
        self.learn_step_counter = 0

        # initialize memory counter
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        # define target_net and eval_net
        self.target_net = NN_model(self.n_features, self.n_actions, self.dueling)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.eval_net = NN_model(self.n_features, self.n_actions, self.dueling)

        if use_pretrained:
            self.eval_net.load_state_dict(os.path.join(self.path_to_para, 'model_para.pth'))
            self.target_net.load_state_dict(os.path.join(self.path_to_para, 'model_para.pth'))

        # define loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)

        self.loss_his = []

    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:
            transition = np.hstack((s, [a, r], s_))
            # replace the old memory with new memory
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        observation = torch.from_numpy(observation).float()

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            action_val = self.eval_net(observation)
            action = action_val.detach().numpy()
            action = np.argmax(action)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            parameter = self.eval_net.state_dict()
            self.target_net.load_state_dict(parameter, strict=True)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
            ISWeights = torch.from_numpy(ISWeights).float()
            batch_memory = torch.from_numpy(batch_memory)
        else:
            # sample batch memory from all memory
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
            batch_memory = torch.from_numpy(batch_memory)

        state = batch_memory[:, :self.n_features].float()
        state_ = batch_memory[:, -self.n_features:].float()
        action = batch_memory[:, self.n_features]
        reward = batch_memory[:, self.n_features + 1].float()

        # compute value of q_eval
        index = action
        mask = build_mask(self.n_actions, index)
        q_eval = torch.masked_select(self.eval_net(state), mask)

        # compute value of q_target
        if self.use_double_q:
            index_t = build_mask(self.n_actions, torch.argmax(self.eval_net(state_), dim=1))
            q_target = reward + self.gamma * torch.masked_select(self.target_net(state_), index_t)
        else:
            q_target = reward + self.gamma * torch.max(self.target_net(state_), 1)[0]

        # compute loss and update the eval_net
        self.optimizer.zero_grad()
        if self.prioritized:
            abs_errors = torch.abs(q_target - q_eval)
            loss = torch.mean(ISWeights * F.mse_loss(q_target, q_eval, reduce=None))
            abs_errors = abs_errors.detach().numpy()
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            loss = self.criterion(q_target, q_eval)

        loss.backward()
        self.optimizer.step()

        #self.loss_his.append(loss)
        self.loss_his.append(loss)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_his)), self.loss_his)
        plt.ylabel('Loss')
        plt.xlabel('training steps')
        plt.show()

    def save_para(self):
        torch.save(self.eval_net.parameters(), os.path.join(self.path_to_para, 'model_para' + '.pth'))






