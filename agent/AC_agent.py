import torch
import torch.optim as optim
import numpy as np
from nn_net.net import Actor, Critic
import torch.nn as nn
import copy
from memorybuffer.ReplayMemory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor_net = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target_net = copy.deepcopy(self.actor_net)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net1 = Critic(state_dim, action_dim).to(device)
        self.critic_net2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_net1 = copy.deepcopy(self.critic_net1)
        self.critic_target_net2 = copy.deepcopy(self.critic_net2)
        self.critic_optimizer = optim.Adam(list(self.critic_net1.parameters()) + list(self.critic_net2.parameters()),
                                           lr=3e-4)

        self.max_action = max_action
        # self.replay_buffer = deque(maxlen=1000000)
        self.replay_buffer = ReplayMemory(1000000)
        self.total_episode = 0

        self.gamma = 0.99
        self.tau = 0.005  # 软更新系数
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2  # actor网络更新频率

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.push((state, action, next_state, reward, done))

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor_net(state)  # [1,3]
        action = action.cpu().data.numpy().flatten()  # (3,)
        return action

    def learn(self, batch_size=100):
        self.total_episode += 1
        # 从经验回放中采样
        # batch = random.sample(self.replay_buffer, batch_size)
        batch = self.replay_buffer.sample(batch_size)
        state, action, next_state, reward, done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(device)     # [100,20]
        action = torch.FloatTensor(np.array(action)).to(device)   # [100,3]
        next_state = torch.FloatTensor(np.array(next_state)).to(device)     # [100,20]
        reward = torch.FloatTensor(np.array(reward)).to(device)        # [100]
        done = torch.FloatTensor(np.array(done)).to(device)           # [100]


        with torch.no_grad():
            # 目标动作 + 噪声
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)  # [100,3]
            next_action = (self.actor_target_net(next_state) + noise).clamp(-self.max_action, self.max_action)  # [100,3]


            # 目标 Q 值
            target_value1 = self.critic_target_net1(next_state, next_action)          # [100,1]
            target_value2 = self.critic_target_net2(next_state, next_action)          # [100,1]
            target_value = reward.unsqueeze(1) + (1 - done).unsqueeze(1) * self.gamma * torch.min(target_value1, target_value2)     # [100,1]


        # 更新 Critic 网络
        current_value1 = self.critic_net1(state, action)
        loss_target1 = nn.MSELoss()(current_value1, target_value)
        current_q2 = self.critic_net2(state, action)           # [100,1]
        loss_target2 = nn.MSELoss()(current_q2, target_value)
        self.critic_optimizer.zero_grad()
        (loss_target1 + loss_target2).backward()

        self.critic_optimizer.step()

        # 每隔 policy_freq 更新 Actor 网络
        if self.total_episode % self.policy_freq == 0:
            actor_loss = -self.critic_net1(state, self.actor_net(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.update_model()

    # 软更新目标网络
    def update_model(self):
        for param, target_param in zip(self.actor_net.parameters(), self.actor_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_net1.parameters(), self.critic_target_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_net2.parameters(), self.critic_target_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor_net.state_dict(), filename + "_actor.pth")
        torch.save(self.critic_net1.state_dict(), filename + "_critic1.pth")
        torch.save(self.critic_net2.state_dict(), filename + "_critic2.pth")

    def load(self, filename):
        self.actor_net.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic_net1.load_state_dict(torch.load(filename + "_critic1.pth"))
        self.critic_net2.load_state_dict(torch.load(filename + "_critic2.pth"))
