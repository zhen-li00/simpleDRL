# coding=utf-8

__author__ = 'l.zhen2281@gmail.com'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import categorical, multivariate_normal

import numpy as np


class ReplayBuffer:     # 经验池，存储轨迹信息
    def __init__(self):
        self.states = []
        self.probs = []     # 概率probability
        self.vals = []      #
        self.actions = []
        self.rewards = []
        self.dones = []     #
        ## self.batch_size = batch_size    # 出现在？（修改至sample函数内，l.zhen2281@gmail.com)

    def sample(self, batch_size):       # 随机采样，方便后续进行随机梯度下降。
        batch_step = np.arange(0, len(self.states), batch_size)    # 0到len(states)，间隔batch_size
        indices = np.arange(len(self.states), dtype=np.int64)       # 0到len(states)，为每个状态编号
        np.random.shuffle(indices)      # 随机打乱编号
        ## 在随机打乱的编号后，划分数据，每份batch_size个
        batches = [indices[i: i+batch_size] for i in batch_step]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches     # 存储了随机采样后的信息

    def push(self, state, action, probs, vals, reward, done):   # 将轨迹保存在memory中
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):    # 清空memory
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class Actor(nn.Module):     # actor网络，输入是状态state，输出是描述动作action的概率分布
    '''
    如果action是连续的，那么actor会输出一个action的均值（维度与action相同），以此构建一个多维正态分布；
    如果action是离散的，那么actor则会输出每一个action的概率，以此构建一个Categorical分布
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.act_dim = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))

        ## action连续:
        # 创建一个多维正态分布
        mean = self.linear3(x)
        # Create our variable for the matrix. Note that I chose 0.5 for stdev arbitrarily.
        cov_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(self.device)
        # Create the covariance matrix
        cov_mat = torch.diag(cov_var).to(self.device)
        dist = multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)

        # ## action离散:
        # # 创建一个类别分布
        # act_probs = F.softmax(self.linear3(x), dim=-1)
        # dist = categorical.Categorical(act_probs)

        return dist


class Critic(nn.Module):    # critic网络，根据当前状态state，得出v值。
    '''critic只输出一个标量（表示输入observation的价值）。'''
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, s):       # 输入维度变成state_dim+action_dim，即将action信息也纳入critic网络中
        x = F.relu(self.linear1(s))
        v = self.linear2(x)
        return v


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]    # 连续动作
        # a_dim = self.env.action_space.n     # 离散动作


        # 构建Actor网络，actor和actor_old
        self.actor = Actor(s_dim, 256, a_dim).to(self.device)

        # 构建Critic网络
        self.critic = Critic(s_dim, 256, 1).to(self.device)

        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # buffer
        self.memory = ReplayBuffer()

    def choose_action(self, s0):        # 多少有点问题
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0).to(self.device)   # 升维度

        dist = self.actor(s0)   # 通过actor网络得到动作的概率分布
        action = dist.sample()  # 从该正态分布中采样得到一个action
        action = torch.squeeze(action)
        # print(action)
        # 使用log_prob()函数，计算这个action的log概率，构造等效的损失函数，用于求策略梯度。
        probs = torch.squeeze(dist.log_prob(action)).item()
        # print(probs)
        '''索命error：ValueError: only one element tensors can be converted to Python scalars
        1、循环式解决，item()一般不适用于取多值，可用.numpy()
        2、用.numpy()，但cuda上变量不能直接转换
        3、使用.detach()'''
        action = [t.item() for t in torch.squeeze(action)]
        # print("action: " + str(action))

        value = torch.squeeze(self.critic(s0)).item()   # 通过critic网络得到状态优势值，A

        return action, probs, value

    def update(self):
        '''
        该部分首先从memory中提取搜集到的轨迹信息，然后计算gae，即advantage，
        接着使用随机梯度下降更新网络，
        最后清除memory以便搜集下一条轨迹信息。
        '''

        for _ in range(self.n_epochs):      # n_epochs?
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, \
            batches = self.memory.sample(batch_size=self.batch_size)      # 获取memory中的一条轨迹信息
            values = vals_arr

            ### compute advantage ###
            ## 优势函数advantage：https://aistudio.baidu.com/aistudio/projectdetail/548292 ##
            # GAE技术：https://zhuanlan.zhihu.com/p/45107835 #
            advantage = np.zeros(len(reward_arr), dtype=np.float32) # 初始化advantage值数组
            for t in range(len(reward_arr)-1):
                discount = 1        # γ*λ
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += \
                        discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            # 归一化advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            advantage = torch.tensor(advantage).to(self.device)

            ###  随机梯度下降SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                actions = actions.to(torch.float32)     # 为什么会数据类型问题？
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()     # 概率比

                # print(advantage[batch], prob_ratio)
                weighted_probs = torch.matmul(advantage[batch].to(torch.float64), prob_ratio)   # 为什么数据类型不同？
                # weighted_probs = advantage[batch] * prob_ratio
                # Clipping the probability ratio，裁剪概率比例
                weighted_clipped_probs = torch.matmul(advantage[batch].to(torch.float64),
                                         torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip))
                # actor网络目标函数（梯度上升）
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()  # 论文Figure 1
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                rewards = advantage[batch] + values[batch]

                critic_loss = ((critic_value-rewards)**2).mean()    # critic网络目标函数，实际回报与期望回报的差平方
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
        self.memory.clear()



