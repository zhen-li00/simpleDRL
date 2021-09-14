# coding: utf-8

__author__ = 'l.zhen2281@gmail.com'


import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(torch.__version__)


class Net(nn.Module):   # 神经网络
    def __init__(self, n_in, n_hid, n_out):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(n_in, n_hid)
        self.layer2 = nn.Linear(n_hid, n_out)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class ReplayBuffer:     # 经验池
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def sample(self, batch_size):   # 随机检索batch_size大小的样本
        return random.sample(self.memory, batch_size)

    def puch(self, *transition):  # 将transition = (state, action, state_next, reward)保存在memory中
        if len(self.memory) == self.capacity:  # 内存已满
            self.memory.pop(0)  # pop()函数用于移除列表中的一个元素（默认最后一个元素）

        self.memory.append(transition)  # 内存未满时添加

    def __len__(self):  # 返回当前memory的长度
        return len(self.memory)


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


        # 创建两个Q网络
        self.main_q_network = Net(self.state_space_dim, 256, self.action_space_dim)
        self.target_q_network = Net(self.state_space_dim, 256, self.action_space_dim)

        print(self.main_q_network)  # 输出网络形状

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=self.lr)   # 优化器（主网络）


        # 创建经验池
        self.buffer = ReplayBuffer(self.capacity)

    def put(self, *transition):
        self.buffer.puch(*transition)

    def act(self, s0, episode):     # 采用ε-贪婪法确定动作
        epsilon = 0.5*(1/(1+episode))
        if epsilon <= random.random():
            s0 = torch.tensor(s0, dtype=torch.float).view(1, -1)
            a0 = torch.argmax(self.main_q_network(s0)).item()
        else:
            a0 = random.randrange(self.action_space_dim)

        return a0

    def learn(self):
        if self.buffer.__len__() < self.batch_size:
            return

        samples = self.buffer.sample(self.batch_size)

        s0, a0, s1, r1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)

        a_max = torch.max(self.main_q_network(s1).detach(), dim=1)[1].view(self.batch_size, -1)

        # print(torch.max(self.main_q_network(s1).detach(), dim=1))
        # print(torch.max(self.main_q_network(s1).detach(), dim=1)[0])
        # print(torch.max(self.main_q_network(s1).detach(), dim=1)[1])
        # print(a_max)
        # print(self.target_q_network(s1))

        target_net_value = self.target_q_network(s1).gather(1, a_max)

        y_true = r1 + self.gamma * target_net_value
        y_pred = self.main_q_network(s0).gather(1, a0)

        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):    # 将目标网络更新到与主网络相同
        # state_dict(): 返回一个包含 Module 实例完整状态的字典
        # load_state_dict(state_dict, strict=True): 从 state_dict 中复制参数和缓冲区到 Module 及其子类中
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())










