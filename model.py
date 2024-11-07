import torch
import torch.nn as nn


hidden_size = 128
'''
ppo网络结构
'''
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层

        # 共享层
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_size),
            nn.ReLU(),
        )

        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, act_dim),
        )

        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        # 动作的对数标准差（用于高斯策略）
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        shared = self.shared_layers(x)
        mean = self.actor(shared)           # [x,y,z]
        value = self.critic(shared).squeeze(-1)   # [int]
        std = torch.exp(self.log_std)            # [x,y,z]
        return mean, std, value





#TD3网络结构


hidden_size1 = 400
hidden_size2 = 300
# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.layer_1 = nn.Linear(state_dim, self.hidden_size1)
        self.layer_2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.layer_3 = nn.Linear(self.hidden_size2, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.relu(self.layer_2(a))
        a = self.max_action * torch.tanh(self.layer_3(a))
        return a

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.layer_1 = nn.Linear(state_dim + action_dim, self.hidden_size1)
        self.layer_2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.layer_3 = nn.Linear(self.hidden_size2, 1)

    def forward(self, state, action):
        q = torch.relu(self.layer_1(torch.cat([state, action], 1)))
        q = torch.relu(self.layer_2(q))
        q = self.layer_3(q)
        return q