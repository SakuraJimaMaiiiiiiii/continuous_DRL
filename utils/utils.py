import numpy as np
import plotly.graph_objects as go



def calculate_path_length(path):
    total_length = 0
    for i in range(1, len(path)):
        current_step = path[i]
        previous_step = path[i - 1]
        step_length = np.sqrt((current_step[0] - previous_step[0]) ** 2 +
                              (current_step[1] - previous_step[1]) ** 2 +
                              (current_step[2] - previous_step[2]) ** 2)
        total_length += step_length
    return total_length


def plot_path(path, env, algorithm):
    # 使用 Plotly 绘制路径
    path_array = np.array(path)

    fig = go.Figure()

    # 添加路径
    fig.add_trace(go.Scatter3d(
        x=path_array[:, 0],
        y=path_array[:, 1],
        z=path_array[:, 2],
        mode='lines+markers',
        line=dict(color='black', width=4),
        marker=dict(size=5),
        name=algorithm
    ))

    # 添加起点和终点
    fig.add_trace(go.Scatter3d(
        x=[env.start[0]], y=[env.start[1]], z=[env.start[2]],
        mode='markers',
        marker=dict(color='green', size=10),
        name='起点'
    ))

    fig.add_trace(go.Scatter3d(
        x=[env.goal[0]], y=[env.goal[1]], z=[env.goal[2]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='终点'
    ))

    # 添加障碍物
    for (x1, y1, z1), (x2, y2, z2) in env.obstacles:
        x = [x1, x2]
        y = [y1, y2]
        z = [z1, z2]
        xx, yy = np.meshgrid(x, y)

        fig.add_trace(go.Surface(
            x=xx, y=yy, z=np.full_like(xx, z1), showscale=False, opacity=0.5, colorscale=[[0, 'blue'], [1, 'blue']]
        ))
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=np.full_like(xx, z2), showscale=False, opacity=0.5, colorscale=[[0, 'blue'], [1, 'blue']]
        ))

        yy, zz = np.meshgrid(y, z)
        fig.add_trace(go.Surface(
            x=np.full_like(yy, x1), y=yy, z=zz, showscale=False, opacity=0.5, colorscale=[[0, 'blue'], [1, 'blue']]
        ))
        fig.add_trace(go.Surface(
            x=np.full_like(yy, x2), y=yy, z=zz, showscale=False, opacity=0.5, colorscale=[[0, 'blue'], [1, 'blue']]
        ))

        xx, zz = np.meshgrid(x, z)
        fig.add_trace(go.Surface(
            x=xx, y=np.full_like(xx, y1), z=zz, showscale=False, opacity=0.5, colorscale=[[0, 'blue'], [1, 'blue']]
        ))
        fig.add_trace(go.Surface(
            x=xx, y=np.full_like(xx, y2), z=zz, showscale=False, opacity=0.5, colorscale=[[0, 'blue'], [1, 'blue']]
        ))

    fig.update_layout(
        title=f"{algorithm} 3D路径图",
        scene=dict(
            xaxis_title='X 轴',
            yaxis_title='Y 轴',
            zaxis_title='Z 轴'
        )
    )

    # 保存为 HTML 文件
    fig.write_html(f"results/html/{algorithm.lower()}.html")
    print(f"{algorithm} 最路径图已保存为 {algorithm.lower()}.html")
