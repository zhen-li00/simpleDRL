# coding=utf-8

__author__ = 'l.zhen2281@gmain.com'


import gym
import matplotlib.pyplot as plt

from ppo import Agent


def plot_rewards(rewards, env='LunarLanderContinuous-v2'):
    plt.title("average learning curve of {} for {}".format('PPO', env))
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards')
    plt.legend()
    plt.show()


if __name__ == '__main__':


    env = gym.make('LunarLanderContinuous-v2')
    env.seed(seed=1)
    env.reset()
    # env.render()

    params = {
        'env': env,
        'actor_lr': 0.0005,
        'critic_lr': 0.0005,
        'n_epochs': 4,  # 问号，不懂
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'policy_clip': 0.2,
        'update_freq': 20,  # frequency of agent update
        'batch_size': 32,    # memory.sample()的超参数
    }

    agent = Agent(**params)
    rewards = []

    print('Start to train !')
    print(f'Env: {agent.env}')
    print(f'State_dim: {agent.env.observation_space.shape[0]}, '
          f'Action_dim: {agent.env.action_space.shape[0]}')

    for episode in range(500):
        s0 = env.reset()
        episode_reward = 0
        running_steps = 0

        done = False
        while not done:
            action, prob, val = agent.choose_action(s0)
            # print(action)
            s1, r1, done, _ = env.step(action)

            running_steps += 1
            episode_reward += r1

            agent.memory.push(s0, action, prob, val, r1, done)

            if running_steps % agent.update_freq == 0:
                agent.update()
            s0 = s1
        rewards.append(episode_reward)
        print(f"Episode:{episode + 1}/{500}, Reward:{episode_reward:.3f}")

    print('Complete training! ')
    plot_rewards(rewards)


