import argparse
import torch
from environment import Environment
from utils import plot_path
from PolicyGradient_agent import PPO,PPOConfig
from AC_agent import TD3
from utils import calculate_path_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def test_td3(rounds=5):
    
    env = Environment(render_mode=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    agent.load("models/td3_agent")

    path_list = [] 

    for episode in range(rounds):
        state = env.reset()
        done = False
        total_reward = 0
        episode_path = []  # 存储路径

        print(f"测试回合 {episode + 1}/{rounds}")
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            episode_path.append(next_state)  # 记录路径
            state = next_state

        path_list.append(episode_path)  # 记录这一回合的路径
        print(episode_path)
        print(f"第 {episode + 1} 回合总奖励：{total_reward:.2f}")

    # 绘制最后一个回合的 3D 路径图
    plot_path(path_list[-1], env,algorithm='td3')


    env.close()



def test_ppo(rounds=10):
    env = Environment(render_mode=True)
    config = PPOConfig()
    agent = PPO(env, config)
    agent.load_model('models/ppo.pth')
    path_list = []  # 存储路径

    for episode in range(rounds):
        state = env.reset()
        done = False
        total_reward = 0
        episode_path = []  

        print(f"测试回合 {episode + 1}/{rounds}")
        
        while not done:
            action, _ = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            episode_path.append(state)

        path_list.append(episode_path) 
        print(f"第 {episode + 1} 回合总奖励：{total_reward:.2f}")


    plot_path(path_list[-1], env, algorithm='ppo')

    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试算法')
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'td3'], default='td3', help='选择要测试的算法 ')
    parser.add_argument('--test_rounds', type=int, default=1, help='测试回合数 ')

    args = parser.parse_args()

    if args.algorithm == 'ppo':
        print('algorith:PPO')
        test_ppo(rounds=args.test_rounds)
    elif args.algorithm == 'td3':
        print('algorith:TD3')
        test_td3(rounds=args.test_rounds)
