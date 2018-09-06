import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


np.random.seed(1)
torch.manual_seed(1)


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

    def __init__(self, n_features, n_actions):
        super(NN_model, self).__init__()
        self.forward1 = nn.Linear(n_features, 20)
        self.forward2 = nn.Linear(20, n_actions)
        self.relu = nn.ReLU(inplace=True)

        self.forward1.weight.data.normal_(0, 0.3)
        self.forward1.bias.data.fill_(0.1)
        self.forward2.weight.data.normal_(0, 0.3)
        self.forward2.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.forward1(x)
        x = self.relu(x)
        x = self.forward2(x)

        return x


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
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize memory counter
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2)) # 6 col (state with 2 + a + r)

        # define target_net and eval_net
        self.target_net = NN_model(self.n_features, self.n_actions)
        self.target_net.eval()
        self.eval_net = NN_model(self.n_features, self.n_actions)
        self.eval_net.train()

        # define loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)

        self.loss_his = []

    def store_transition(self, s, a, r, s_):
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

        # compute loss and update the eval_net
        self.optimizer.zero_grad()

        # compute value of q_target
        q_target = reward + self.gamma * torch.max(self.target_net(state_), 1)[0]

        # compute value of q_eval
        index = action
        mask = build_mask(self.n_actions, index)
        q_eval = torch.masked_select(self.eval_net(state), mask)

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




