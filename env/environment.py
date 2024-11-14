import gym
from gym import spaces
import numpy as np
import matplotlib
from env.obstacles import env1, env2
from args import basic_set
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams


args = basic_set()


class Environment(gym.Env):
    def __init__(self, render_mode=False):
        super(Environment, self).__init__()
        self.grid_size = args.grid_size  # 地图尺寸
        self.start = np.array([1, 1, 1], dtype=np.float32)
        self.goal = np.array([self.grid_size - 1, self.grid_size - 1, self.grid_size - 1], dtype=np.float32)
        # self.start = np.array([3, self.grid_size // 2, 3], dtype=np.float32)
        # self.goal = np.array([self.grid_size - 3, self.grid_size // 2, self.grid_size - 3], dtype=np.float32)
        self.state = self.start.copy()
        self.delta = args.delta  # 达到目标的距离阈值
        self.render_mode = render_mode
        self.max_steps = args.max_step
        self.steps_taken = 0
        self.sampletime = args.sampletime  # 路径采样点分辨率 0.1

        # 障碍物
        self.obstacles = env2
        # self.obstacles = env1


        # 动作空间范围-2.0, 2.0 shape(3,)为三维向量 表示x,y,z上的动作
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(3,),
                                       dtype=np.float32)  # space.Box为gym库中的连续空间   shape: (3,)  shape[0] =3

        # print("Action space lower bounds:", self.action_space.low)  # 输出 [-2.0, -2.0, -2.0]
        # print("Action space upper bounds:", self.action_space.high)  # 输出 [2.0, 2.0, 2.0]
        # print("Action space shape:", self.action_space.shape)  # 输出 (3,)
        # print("Action space dtype:", self.action_space.dtype)  # 输出 float32

        # 观测空间
        self.num_sensors = args.num_sensors  # 模拟距离传感器       12
        self.sensor_range = args.sensor_range  # 传感器的探测范围 每个传感器的最大检测范围  10
        self.max_distance = np.sqrt(3 * (self.grid_size) ** 2)  # 空间立方体的对角线长度 空间的最远距离
        # 状态表示，归一化向量和标量距离
        # 第一个[-1.0]*3 [1.0]*3 代表相对位置的三个维度(x,y,z)
        # 第二个[0,0]*1 [1.0]*1 代表归一化距离的上下限
        # 第三个[-1,0]*3 [1.0]*3 代表朝目标方向的向量
        # 第四个 [0][1]为标量距离的上限
        # 第五个 [0]*self.num_sensors [1]*self.num_sensors 表示传感器读数的下限

        obs_low = np.array([-1.0] * 3 + [0.0] * 1 + [-1.0] * 3 + [0.0] + [0.0] * self.num_sensors, dtype=np.float32)
        obs_high = np.array([1.0] * 3 + [1.0] * 1 + [1.0] * 3 + [1.0] + [1.0] * self.num_sensors, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)  # shape: (20,)

        # 传感器方向
        self.sensor_directions = np.array([
            [1, 0, 0],  # 正 x 方向
            [-1, 0, 0],  # 负 x 方向
            [0, 1, 0],  # 正 y 方向
            [0, -1, 0],  # 负 y 方向
            [0, 0, 1],  # 正 z 方向
            [0, 0, -1],  # 负 z 方向
            [1, 1, 0],  # x 和 y 正方向
            [1, -1, 0],  # x 正，y 负
            [-1, 1, 0],  # x 负，y 正
            [-1, -1, 0],  # x 和 y 负方向
            [0, 1, 1],  # y 和 z 正方向
            [0, -1, -1],  # y 和 z 负方向
        ], dtype=np.float32)

        # 归一化传感器方向   np.linalg.norm(vec) 计算向量的范数(长度) len = √x2+y2+z2
        self.sensor_directions = np.array([vec / np.linalg.norm(vec) for vec in self.sensor_directions])
        # [1.          0.          0.]
        # [-1.         0.          0.]
        # [0.          1.          0.]
        # [0.        - 1.          0.]
        # [0.          0.          1.]
        # [0.          0.        - 1.]
        # [0.70710677  0.70710677  0.]
        # [0.70710677 -0.70710677  0.]
        # [-0.7071067  0.70710677  0.]
        # [-0.7071067 -0.70710677  0.]
        # [0.          0.70710677  0.70710677]
        # [0.         -0.70710677 -0.70710677]

        self.fig = None
        self.ax = None
        self.agent_plot = None

        # 智能体路径
        self.path = []

        rcParams['font.sans-serif'] = ['SimHei']
        rcParams['axes.unicode_minus'] = False

    def reset(self):
        # 重置位置、步数和路径
        self.state = self.start.copy()
        self.steps_taken = 0
        self.path = [self.state.copy()]
        if self.render_mode:
            self._init_render()
        current_distance = np.linalg.norm(self.state - self.goal)  # len = √(x1-x2)2+(y1-y2)2+(z1-z2)2     "33.941124"
        relative_position = (self.goal - self.state) / self.grid_size  # 相对位置 len/grid_size   "[0.8 0.  0.8]"
        normalized_distance = current_distance / self.max_distance  # 归一化距离   len/√3grid_size  范围：(0,1)   "0.653197235209715"
        direction_to_goal = (self.goal - self.state) / (np.linalg.norm(
            self.goal - self.state) + 1e-8)  # 归一化方向向量  position -> goal      "[0.7071068 0.        0.7071068]"
        scalar_distance = current_distance / self.max_distance  # 标量距离   与normalized_distance数值完全相同 作为标量距离被单独储存   " 0.653197235209715"
        sensor_readings = self._get_sensor_readings() / self.sensor_range  # 传感器读数  [0.7777778 0.3333333 1. 1. 1. 0.3333333 1. 1. 0.44444448 0.44444448 1. 0.44444448]

        # 构建观测
        observation = np.concatenate(
            (relative_position, [normalized_distance], direction_to_goal, [scalar_distance], sensor_readings))
        # observation = [relative_position(3values),normalized_distance(1value),direction_to_goal(3values),scalar_distance(1value),sensor_readings(self.num_sensorsvalues)]
        # len(observation) = 20
        return observation

    def step(self, action):
        # 确保动作在动作空间内

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 更新步数
        self.steps_taken += 1
        action = np.squeeze(action)
        # 记录之前的位置和距离
        previous_state = self.state.copy()
        previous_distance = np.linalg.norm(self.state - self.goal)

        '''
        位置更新
        '''
        # 用动作更新位置
        self.state += action
        self.state = np.clip(self.state, 0, self.grid_size - 1)

        # 碰撞标志
        is_collision = False

        # 沿着移动路径进行碰撞检测
        movement_vector = self.state - previous_state  # 移动向量
        movement_length = np.linalg.norm(movement_vector)  # 移动长度

        if movement_length > 0:
            num_samples = int(
                np.ceil(movement_length / self.sampletime))  # 0.1:分辨率  num_samples为采样次数保证 state和next_state之间没有障碍
            sample_interval = np.linspace(0, 1, num_samples + 1)  # 采样间隔  [0~1]
            for i, t in enumerate(sample_interval):
                inter_position = previous_state + t * movement_vector
                in_obstacle = False
                for (x1, y1, z1), (x2, y2, z2) in self.obstacles:
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    z_min, z_max = min(z1, z2), max(z1, z2)
                    if (x_min <= inter_position[0] <= x_max and
                            y_min <= inter_position[1] <= y_max and
                            z_min <= inter_position[2] <= z_max):
                        in_obstacle = True
                        break
                if in_obstacle:
                    is_collision = True
                    # 将位置设置为碰撞前的最后安全位置
                    if i == 0:  # 在起始点本来就在就发生碰撞，保持原位
                        self.state = previous_state
                    else:
                        # 上一个安全位置
                        self.state = previous_state + sample_interval[i - 1] * movement_vector
                    break
        else:
            # 没有移动
            self.state = previous_state

        self.path.append(self.state.copy())

        ''''
        奖励设计
        '''
        # 当前位置与目标的距离
        current_distance = np.linalg.norm(self.state - self.goal)

        # 奖励函数
        # 潜在奖励使用距离的差值作为潜在函数
        distance_reward = (previous_distance - current_distance) / self.max_distance  # 距离离得越近 奖励越大

        # 密集奖励鼓励朝向目标的移动方向
        direction = (self.state - previous_state) / (np.linalg.norm(self.state - previous_state) + 1e-8)
        desired_direction = (self.goal - self.state) / (np.linalg.norm(self.goal - self.state) + 1e-8)
        direction_reward = np.dot(direction, desired_direction)  # <1

        # 总奖励
        reward = args.dis_c * distance_reward + args.dir_c * direction_reward  # <1          ini: 10   0.5

        if is_collision:
            reward -= args.hit_r  # 碰撞惩罚    ini:1

        # 检查是否到达目标
        if current_distance < self.delta:
            reward += args.arr_r  # 到达目标的奖励      ini:5
            done = True
        else:
            done = False

        # 最大步数限制
        if self.steps_taken >= self.max_steps:
            done = True

        # 获取传感器
        sensor_readings = self._get_sensor_readings() / self.sensor_range

        # 避障惩罚
        min_sensor_reading = np.min(sensor_readings)
        if min_sensor_reading < 0.2:
            reward -= (0.2 - min_sensor_reading) * args.avo_c  # 离障碍物太近的惩罚  # 1.0

        # 时间惩罚
        reward -= args.time_r  # 0.01

        info = {'distance_to_goal': current_distance / self.max_distance, 'collision': is_collision, 'path': self.path}
        # 计算相对位置、归一化距离、方向向量和标量距离
        relative_position = (self.goal - self.state) / self.grid_size
        normalized_distance = current_distance / self.max_distance
        direction_to_goal = (self.goal - self.state) / (np.linalg.norm(self.goal - self.state) + 1e-8)
        scalar_distance = current_distance / self.max_distance

        # 构建观测
        next_state = np.concatenate(
            (relative_position, [normalized_distance], direction_to_goal, [scalar_distance], sensor_readings))

        return next_state, reward, done, info

    def _get_sensor_readings(self):
        # 传感器读数函数
        readings = []
        for direction_vetor in self.sensor_directions:
            min_distance = self.sensor_range  # 将探测到有障碍物的最远距离设置为传感器的感应最远距离 返回时该方向距离障碍物距离为sensor_range
            for i in np.linspace(0, self.sensor_range, num=10):
                detect_pos = self.state + direction_vetor * i  # 检查探测位置是否越界
                if (detect_pos < 0).any() or (detect_pos > self.grid_size).any():
                    min_distance = i
                    break  # 超出边界
                is_obstacle = False
                for (x1, y1, z1), (x2, y2, z2) in self.obstacles:
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    z_min, z_max = min(z1, z2), max(z1, z2)
                    if (x_min <= detect_pos[0] <= x_max and
                            y_min <= detect_pos[1] <= y_max and
                            z_min <= detect_pos[2] <= z_max):
                        min_distance = i
                        is_obstacle = True
                        break
                if is_obstacle:
                    break
            readings.append(min_distance)
        return np.array(readings, dtype=np.float32)

    def _init_render(self):

        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_zlim(0, self.grid_size)
            self.ax.set_xlabel('X 轴')
            self.ax.set_ylabel('Y 轴')
            self.ax.set_zlabel('Z 轴')

            self.ax.scatter(*self.start, color='green', s=50, label='start')
            self.ax.scatter(*self.goal, color='red', s=50, label='final')

            for (x1, y1, z1), (x2, y2, z2) in self.obstacles:
                self._draw_obstacle(x1, y1, z1, x2, y2, z2)

            self.ax.legend()

    def _draw_obstacle(self, x1, y1, z1, x2, y2, z2):
        # 障碍物
        x = [x1, x2]
        y = [y1, y2]
        z = [z1, z2]
        xx, yy = np.meshgrid(x, y)

        self.ax.plot_surface(xx, yy, np.full_like(xx, z1), color='blue', alpha=0.5)
        self.ax.plot_surface(xx, yy, np.full_like(xx, z2), color='blue', alpha=0.5)

        yy, zz = np.meshgrid(y, z)
        self.ax.plot_surface(np.full_like(yy, x1), yy, zz, color='blue', alpha=0.5)
        self.ax.plot_surface(np.full_like(yy, x2), yy, zz, color='blue', alpha=0.5)

        xx, zz = np.meshgrid(x, z)
        self.ax.plot_surface(xx, np.full_like(xx, y1), zz, color='blue', alpha=0.5)
        self.ax.plot_surface(xx, np.full_like(xx, y2), zz, color='blue', alpha=0.5)

    def render(self, mode='human'):

        if not self.render_mode:
            return

        if self.fig is None or self.ax is None:
            self._init_render()
        else:
            self.ax.clear()
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_zlim(0, self.grid_size)
            self.ax.set_xlabel('X 轴')
            self.ax.set_ylabel('Y 轴')
            self.ax.set_zlabel('Z 轴')

            self.ax.scatter(*self.start, color='green', s=100, label='起点')
            self.ax.scatter(*self.goal, color='red', s=100, label='终点')

            for (x1, y1, z1), (x2, y2, z2) in self.obstacles:
                self._draw_obstacle(x1, y1, z1, x2, y2, z2)

            path_array = np.array(self.path)
            self.ax.plot3D(path_array[:, 0], path_array[:, 1], path_array[:, 2], color='black', label='路径')
            self.ax.scatter(*self.state, color='black', s=60)

            self.ax.legend()

        plt.draw()
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == '__main__':

    env = Environment(render_mode=True)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()
