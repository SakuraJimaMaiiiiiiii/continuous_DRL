import os
import numpy as np
import torch
from env.environment import Environment
from utils.utils import calculate_path_length
from agent.AC_agent import TD3
from args import basic_set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = basic_set()
actor_net_dir = r'models/td3model/actor'
critic_net1_dir = r'models/td3model/critic1'
critic_net2_dir = r'models/td3model/critic2'
os.makedirs(actor_net_dir, exist_ok=True)
os.makedirs(critic_net1_dir, exist_ok=True)
os.makedirs(critic_net2_dir, exist_ok=True)


# 训练 TD3
def train_td3():
    env = Environment()
    state_dim = env.observation_space.shape[0]            # 20
    action_dim = env.action_space.shape[0]                # 3
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    exploration_noise = args.exploration_noise            # 0.1
    episodes = args.td3_episodes                    # 5000
    batch_size = args.td3_batchsize                 # 100
    warmup_steps = args.warmup_steps                #5000
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

        # print(f"Episode: {episode}, Reward: {episode_reward},总路程：{calculate_path_length(path_len)}")

        average_return = np.mean(episode_reward)
        if episode % 50 == 0:
            # agent.save("models/td3_agent")
            print(f"第 {episode + 1} 回合 avg reward：{average_return:.2f}，总步数：{steps}，总路程：{calculate_path_length(path_len)}")
            torch.save(agent.actor_net.state_dict(), f'{actor_net_dir}/actor_net_step{episode + 1}.pth')
            torch.save(agent.critic_net1.state_dict(), f'{critic_net1_dir}/critic_net1_step{episode + 1}.pth')
            torch.save(agent.critic_net2.state_dict(), f'{critic_net2_dir}/critic_net2_step{episode + 1}.pth')

    env.close()



if __name__ == '__main__':
    print('algorithm: td3')
    train_td3()
