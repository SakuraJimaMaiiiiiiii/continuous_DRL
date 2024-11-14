import os
import numpy as np
import torch
from env.environment import Environment
from utils.utils import calculate_path_length
from agent.PolicyGradient_agent import PPO, PPOConfig
from args import basic_set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = basic_set()
ActorCritic_net_dir = r'models/PPOmodel'
os.makedirs(ActorCritic_net_dir, exist_ok=True)



# 训练 PPO
def train_ppo():
    env = Environment(render_mode=False)
    config = PPOConfig()
    agent = PPO(env, config)

    total_steps = 0
    max_episodes = args.ppo_episodes  # 1000
    batch_size = config.batch_size

    for episode in range(max_episodes):
        state = env.reset()
        path_len = []

        episode_rewards = []              # 每回合的累计奖励
        state_buffer = []
        action_buffer = []
        advantage_buffer = []
        targetValue_buffer = []                # 每步的折扣回报
        logp_buffer = []                  # 每个动作的对数概率
        rewards = []
        values = []
        dones = []

        ep_len = 0
        ep_reward = 0

        while True:
            action, log_prob = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            state_buffer.append(state)
            action_buffer.append(action)
            rewards.append(reward)
            dones.append(done)
            logp_buffer.append(log_prob)

            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, _, value = agent.ActorCritic_net(state_tensor)
            values.append(value.item())

            state = next_state

            if done or ep_len == config.episodes:
                next_value = 0 if done else agent.ActorCritic_net(torch.tensor(state, dtype=torch.float32).to(device))[2].item()
                values.append(next_value)

                advantages = agent.compute_GAE(rewards, values, dones)
                # reward + γQ(next_state)
                targetValue = [adv + val for adv, val in zip(advantages, values[:-1])]     # 截取values列表中的最后一个 因为不需要最后一个next_value

                advantage_buffer.extend(advantages)               # 由于advantages为列表 所以用extend而不是append
                targetValue_buffer.extend(targetValue)
                path_len.extend(info['path'])
                total_steps += ep_len

                episode_rewards.append(ep_reward)

                ep_reward = 0
                ep_len = 0
                rewards = []
                values = []
                dones = []

                state = env.reset()

                if total_steps >= batch_size:
                    break

        state_buffer = np.array(state_buffer)
        action_buffer = np.array(action_buffer)
        advantage_buffer = np.array(advantage_buffer)
        targetValue_buffer = np.array(targetValue_buffer)
        logp_buffer = np.array(logp_buffer)


        advantage_buffer = (advantage_buffer - advantage_buffer.mean()) / (advantage_buffer.std() + 1e-8)        # 归一化处理 使训练更稳定

        agent.learn(state_buffer, action_buffer, advantage_buffer, targetValue_buffer, logp_buffer)

        average_return = np.mean(episode_rewards)

        total_steps = 0
        if (episode+1) % 10 == 0:
            print(f"第 {episode+1} 回合 avg reward：{average_return:.2f}，总步数：{total_steps}，总路程：{calculate_path_length(path_len)}")
            torch.save(agent.ActorCritic_net.state_dict(), f'{ActorCritic_net_dir}/ActorCritic_net_step{episode+1}.pth')
    # agent.save_model('models/PPOmodel/ppo.pth')

    env.close()



if __name__ == '__main__':
    print('algorithm:  ppo')
    train_ppo()
