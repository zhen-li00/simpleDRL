# coding :utf-8

__author__ = 'zhenhang.sun@gmail.com'
__version__ = '1.0.0'

import gym
import matplotlib.pyplot as plt

from ddpg import Agent


def plot_result(score, mean):
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.plot(score)
    plt.plot(mean)
    plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.text(len(mean) - 1, mean[-1], str(mean[-1]))
    plt.show()


if __name__ == '__main__':

    env = gym.make('gpu_version')
    env.reset()
    env.render()

    params = {
        'env': env,
        'gamma': 0.9,
        'actor_lr': 0.001,
        'critic_lr': 0.002,
        'tau': 0.01,    # 软更新参数
        'capacity': 10000,
        'batch_size': 32,
    }

    agent = Agent(**params)

    score = []
    step_mean = []

    for episode in range(300):
        s0 = env.reset()
        episode_reward = 0

        for step in range(200):
            env.render()
            a0 = agent.act(s0)
            # print(a0)
            s1, r1, done, _ = env.step(a0)
            agent.put(s0, a0, r1, s1)

            episode_reward += r1
            s0 = s1

            agent.learn()

        score.append(episode_reward)
        step_mean.append(episode_reward/200)

        print(episode, ': ', episode_reward)

    plot_result(score, step_mean)
