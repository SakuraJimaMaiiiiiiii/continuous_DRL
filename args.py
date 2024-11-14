import argparse


def basic_set():
    parser = argparse.ArgumentParser()

    # env setting
    parser.add_argument('--grid_size', type=int, default=30, help='环境尺寸')
    parser.add_argument('--delta', type=int, default=1, help='到达目标的距离阈值')
    parser.add_argument('--max_step', type=int, default=200, help='最大步数')
    parser.add_argument('--sampletime', type=int, default=0.1, help='路径采样点分辨率')
    parser.add_argument('--num_sensors', type=int, default=12, help='模拟距离传感器')
    parser.add_argument('--sensor_range', type=int, default=10, help='传感器的探测范围 每个传感器的最大检测范围')

    # reward setting
    parser.add_argument('--dis_c', type=int, default=10, help='距离差值奖励系数')
    parser.add_argument('--dir_c', type=int, default=0.5, help='方向奖励鼓励系数')
    parser.add_argument('--hit_r', type=int, default=1, help='障碍物惩罚')
    parser.add_argument('--arr_r', type=int, default=5, help='到达奖励')
    parser.add_argument('--avo_c', type=int, default=1, help='避障惩罚系数')
    parser.add_argument('--time_r', type=int, default=0.01, help='时间惩罚')

    # train_ppo setting
    parser.add_argument('--ppo_episodes', type=int, default=1000, help='最大训练回合')


    # train_td3 setting
    parser.add_argument('--td3_episodes', type=int, default=5000, help='最大训练回合')
    parser.add_argument('--exploration_noise', type=int, default=0.1, help='训练噪声')
    parser.add_argument('--td3_batchsize', type=int, default=100, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='warmup_steps')




    # train
    parser.add_argument('--train_algorithm', type=str, choices=['ppo','td3'], default='td3', help='训练算法')

    args = parser.parse_args()
    return args