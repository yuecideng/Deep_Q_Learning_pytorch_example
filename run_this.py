from maze_env import Maze
#from DQN_tensorflow import DeepQNetwork
from DQN_pytorch import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np


def run_maze():
    step = 0
    total_reward_list = []
    for episode in range(300):
        # initial observation
        observation = env.reset()
        total_reward = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            print action
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            total_reward += reward

            # break while loop when end of this episode
            if done:
                break
            step += 1

        total_reward_list.append(total_reward)

    plt.plot(np.arange(len(total_reward_list)), total_reward_list)
    plt.ylabel('Total_reward')
    plt.xlabel('training epoch')
    plt.show()

    # end of game
    print('game over')

    env.destroy()
    print 'over'

if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_loss()
