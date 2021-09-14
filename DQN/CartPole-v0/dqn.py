# coding: utf-8

__author__ = 'zhenhang.sun@gmail.com'
__version__ = '1.0.0'

import gym
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):       # 构建一个神经网络
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):   # 前向传播
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Agent(object):
    def __init__(self, **kwargs):       # kwargs: keyword arguments. ## https://www.jianshu.com/p/0ed914608a2c
        for key, value in kwargs.items():
            setattr(self, key, value)       # 指定对象的指定属性的值。
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim)
        print(self.eval_net)  # 输出网络形状

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.buffer = []
        self.steps = 0
        
    def act(self, s0):      # 采取动作
        self.steps += 1
        # ？？？（贪婪？）
        epsi = self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = torch.tensor(s0, dtype=torch.float).view(1, -1)    # view()函数：改变tensor的形状
            a0 = torch.argmax(self.eval_net(s0)).item()
        return a0

    def put(self, *transition):    # 将transition = (state, action, state_next, reward)保存在memory中
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)    # pop()函数用于移除列表中的一个元素（默认最后一个元素）
        self.buffer.append(transition)    # 内存未满时添加
        
    def learn(self):
        if (len(self.buffer)) < self.batch_size:    # 经验池大小小于小批量数据时不执行任何操作
            return
        
        samples = random.sample(self.buffer, self.batch_size)    # 从经验池获取小批量数据

        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        y_true = r1 + self.gamma * torch.max(self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)    # gather获取相应Q值

        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

