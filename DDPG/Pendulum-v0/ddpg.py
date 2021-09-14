# coding :utf-8

__author__ = 'zhenhang.sun@gmail.com'
__version__ = '1.0.0'

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):     # actor网络，学习高回报策略，根据状态s采取行动
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))     # tanh(), 输出(-1, 1) ?  映射在哪里
        x = 2 * x       # 映射部分（l.zhen2281@gmail.com 修改）

        return x


class Critic(nn.Module):        # critic网络，对当前策略进行评价，学习当前策略的值函数
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)    # cat: concatenate拼接两个tensor，按维数1（列）拼接
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        # Actor网络
        self.actor = Actor(s_dim, 256, a_dim)
        self.actor_target = Actor(s_dim, 256, a_dim)
        # Critic网络
        self.critic = Critic(s_dim+a_dim, 256, a_dim)
        self.critic_target = Critic(s_dim+a_dim, 256, a_dim)
        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = []        # 可考虑定义replay buffer类

        # 网络初始化到相同参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)   # unsqueeze(arg): 表示在第arg维增加一个维度值为1的维度。
        a0 = self.actor(s0).squeeze(0).detach().numpy()         # squeeze(arg)表示第arg维的维度值为1，则去掉该维度。否则tensor不变。
        return a0
    
    def put(self, *transition): 
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        def critic_learn():
            '''　3. Critic当前网络：负责价值网络参数w的迭代更新，负责计算负责计算当前Q值Q(S,A,w)。目标Q值yi=R+γQ′(S′,A′,w′)
            　　4. Critic目标网络：负责计算目标Q值中的Q′(S′,A′,w′)部分。网络参数w′定期从w复制。'''
            a1 = self.actor_target(s1).detach()

            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            ''' 1. Actor当前网络：负责策略网络参数θ的迭代更新，负责根据当前状态S选择当前动作A，用于和环境交互生成S′,R。
            　 2. Actor目标网络：负责根据经验回放池中采样的下一状态S′选择最优下一动作A′。网络参数θ′定期从θ复制。 '''

            loss = -torch.mean( self.critic(s0, self.actor(s0)) )       # actor网络loss的定义：使评价得分变高（负号），梯度上升
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):      # 软更新
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
                                           
                                           
  
