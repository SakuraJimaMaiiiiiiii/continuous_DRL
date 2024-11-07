import argparse
import numpy as np
import torch
from environment import Environment
from utils import calculate_path_length
from PolicyGradient_agent import PPO,PPOConfig
from AC_agent import TD3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



ActorCritic_net_dir = r'models/PPOmodel'

# 训练 PPO
def train_ppo():
    env = Environment(render_mode=False)
    config = PPOConfig()
    agent = PPO(env, config)

    total_steps = 0
    max_episodes = 1000
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


        print(f"第 {episode+1} 回合 avg reward：{average_return:.2f}，总步数：{total_steps}，总路程：{calculate_path_length(path_len)}")
        total_steps = 0
        torch.save(agent.ActorCritic_net.state_dict(), f'{ActorCritic_net_dir}/ActorCritic_net_step_{episode + 1}.pth')

    agent.save_model('models/ppo.pth')
    env.close()



# 训练 TD3
def train_td3():
    env = Environment()
    state_dim = env.observation_space.shape[0]            # 20
    action_dim = env.action_space.shape[0]                # 3
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    exploration_noise = 0.1
    episodes = 1000
    batch_size = 100
    warmup_steps = 5000
    steps = 0


    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        path_len = []

        while not done:
            if steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)
                action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)

            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            steps += 1
            path_len.extend(info['path'])

            if len(agent.replay_buffer) > batch_size:
                agent.learn(batch_size)

        exploration_noise = max(0.01, exploration_noise * 0.995)

        print(f"Episode: {episode}, Reward: {episode_reward},总路程：{calculate_path_length(path_len)}")

        if episode % 50 == 0:
            agent.save("models/td3_agent")

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'td3'], default='td3', help='训练算法 ')
    args = parser.parse_args()

    if args.algorithm == 'ppo':
        print('algorithm: ',args.algorithm)
        train_ppo()
    else:
        train_td3()
        print('algorithm: ',args.algorithm)