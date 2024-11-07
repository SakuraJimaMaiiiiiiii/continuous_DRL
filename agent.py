import torch
import torch.optim as optim
import numpy as np
from model import ActorCritic,Actor,Critic
import random
import torch.nn as nn
import copy
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOConfig:
    def __init__(self):
        self.gamma = 0.99  # 折扣因子
        self.lmbda = 0.95    # GAE lambda 缩放系数
        self.clip_ratio = 0.1  # PPO剪辑范围参数
        self.lr = 3e-4     # 学习率
        self.train_steps = 80  # 每次更新的训练迭代次数
        self.target_kl = 0.05  # KL散度目标
        self.batch_size = 5000  # 训练批次大小
        self.minibatch_size = 64  # minibatch
        self.episodes = 200  # episode长度

# PPO
class PPO:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        state_dim = env.observation_space.shape[0]    # 20
        action_dim = env.action_space.shape[0]        # 3


        self.ActorCritic_net = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.ActorCritic_net.parameters(), lr=config.lr)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, std, _ = self.ActorCritic_net(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()                                 # 取值在mean±3*std
        log_prob = dist.log_prob(action).sum(axis=-1)          # 用于概率比计算
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def compute_GAE(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = values[-1]
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            td_delta = rewards[step] + self.config.gamma * next_value * mask - values[step]
            gae = td_delta + self.config.gamma * self.config.lmbda * gae * mask
            advantages.insert(0, gae)            # advantages正序
            next_value = values[step]
        return advantages

    def learn(self, state_buffer, action_buffer, advantage_buff, targetValue_buffer, logp_buffer):
        state_buffer = torch.tensor(state_buffer, dtype=torch.float32).to(device)
        action_buffer = torch.tensor(action_buffer, dtype=torch.float32).to(device)
        advantage_buff = torch.tensor(advantage_buff, dtype=torch.float32).to(device)
        targetValue_buffer = torch.tensor(targetValue_buffer, dtype=torch.float32).to(device)
        logp_buffer = torch.tensor(logp_buffer, dtype=torch.float32).to(device)               # 动作的旧对数概率

        dataset_size = state_buffer.shape[0]      # 5000
        total_approx_kl = 0.0                                         # 初始化kl散度

        for _ in range(self.config.train_steps):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.config.minibatch_size):         # [0,dataset_size]按minibatch进行取样
                end = start + self.config.minibatch_size
                minibatch_indices = indices[start:end]

                state_batch = state_buffer[minibatch_indices]
                action_batch = action_buffer[minibatch_indices]
                advantage_batch = advantage_buff[minibatch_indices]
                state_target = targetValue_buffer[minibatch_indices]                        # state_target为reward + γQ(next_state)
                logp_old_batch = logp_buffer[minibatch_indices]

                mean, std, state_value = self.ActorCritic_net(state_batch)             # state_value为神经网络计算出的值
                dist = torch.distributions.Normal(mean, std)
                log_new_batch = dist.log_prob(action_batch).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()           # 正态分布的熵


                ratio = torch.exp(log_new_batch - logp_old_batch)            # 新旧策略之间的比例
                # PPO目标函数
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = ((state_value - state_target) ** 2).mean()

                # 熵奖励权重
                loss = policy_loss + 0.5 * value_loss - 0.02 * entropy

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 累加KL散度
                approx_kl = (logp_old_batch - log_new_batch).mean().item()
                total_approx_kl += approx_kl

            mean_approx_kl = total_approx_kl / (dataset_size / self.config.minibatch_size)
            if mean_approx_kl > 1.5 * self.config.target_kl:
                break

    def save_model(self, filepath):
        torch.save(self.ActorCritic_net.state_dict(), filepath)
        print(f"模型已保存")

    def load_model(self, filepath):
        self.ActorCritic_net.load_state_dict(torch.load(filepath, map_location=device))
        print(f"模型已加载")


class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor_net = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target_net = copy.deepcopy(self.actor_net)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net1 = Critic(state_dim, action_dim).to(device)
        self.critic_net2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_net1 = copy.deepcopy(self.critic_net1)
        self.critic_target_net2 = copy.deepcopy(self.critic_net2)
        self.critic_optimizer = optim.Adam(list(self.critic_net1.parameters()) + list(self.critic_net2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.replay_buffer = deque(maxlen=1000000)
        self.total_eposide = 0

        self.gamma = 0.99
        self.tau = 0.005                # 软更新系数
        self.policy_noise =0.2
        self.noise_clip = 0.5
        self.policy_freq = 2            # actor网络更新频率

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor_net(state)                   # [1,3]
        action = action.cpu().data.numpy().flatten()     # (3,)
        return action


    def learn(self, batch_size=100):
        self.total_eposide += 1
        # 从经验回放中采样
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        reward = torch.FloatTensor(np.array(reward)).to(device)
        done = torch.FloatTensor(np.array(done)).to(device)

        with torch.no_grad():
            # 目标动作 + 噪声
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target_net(next_state) + noise).clamp(-self.max_action, self.max_action)

            # 目标 Q 值
            target_q1 = self.critic_target_net1(next_state, next_action)
            target_q2 = self.critic_target_net2(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * torch.min(target_q1, target_q2)

        # 更新 Critic 网络
        current_q1 = self.critic_net1(state, action)
        loss_q1 = nn.MSELoss()(current_q1, target_q)
        current_q2 = self.critic_net2(state, action)
        loss_q2 = nn.MSELoss()(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        (loss_q1 + loss_q2).backward()
        self.critic_optimizer.step()

        # 每隔 policy_freq 更新 Actor 网络
        if self.total_eposide % self.policy_freq == 0:
            actor_loss = -self.critic_net1(state, self.actor_net(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
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