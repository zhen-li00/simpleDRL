# coding: utf-8


import gym
import matplotlib.pyplot as plt

from ddqn import Agent


def plot(score, mean):
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score)
    plt.plot(mean)
    plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.text(len(mean) - 1, mean[-1], str(mean[-1]))
    plt.show()


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')

    params = {
        'gamma': 0.8,  # 折扣率
        'lr': 0.001,
        'capacity': 10000,  # memory最大长度
        'batch_size': 64,
        'state_space_dim': env.observation_space.shape[0],  # 状态变量：
        'action_space_dim': env.action_space.n,  # 动作类型：
    }
    agent = Agent(**params)

    score = []
    mean = []

    for episode in range(200):
        print('----- episode: ' + str(episode) + ' -----')
        s0 = env.reset()
        total_reward = 1
        while True:
            env.render()
            a0 = agent.act(s0, episode)
            s1, r1, done, _ = env.step(a0)

            if done:
                r1 = -1

            agent.put(s0, a0, s1, r1)

            if done:
                score.append(total_reward)
                print('total_reward: ' + str(total_reward))
                mean.append(sum(score[-100:]) / 100)
                print('mean: ' + str(sum(score[-100:]) / 100))
                break

            total_reward += r1
            s0 = s1
            agent.learn()

        if episode % 5 == 0:    # replace_target_freq: 5   ##更新目标网络的频率
            agent.update_target_network()

    env.close()

    # print(score)
    # print(mean)
    plot(score, mean)

