import torch
import torch.optim as optim
import numpy as np
from nn_net.net import ActorCritic

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

