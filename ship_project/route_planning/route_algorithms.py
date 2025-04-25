import numpy as np
import math
import heapq
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from torch.optim import Adam
from tqdm import tqdm

# 模拟海洋环境地图
# 这里简化处理，实际应用中可能需要使用真实的海图数据
GRID_SIZE = 100
MAP_SIZE = (GRID_SIZE, GRID_SIZE)
OBSTACLES = set()  # 障碍物位置集合

# 初始化一些随机障碍物
for _ in range(GRID_SIZE * 5):
    x = random.randint(0, GRID_SIZE - 1)
    y = random.randint(0, GRID_SIZE - 1)
    OBSTACLES.add((x, y))

# A*算法实现
def heuristic(a, b):
    """计算两点之间的距离
    如果没有障碍物，使用曼哈顿距离以鼓励直线路径
    否则使用欧几里得距离以允许对角线移动
    """
    if not OBSTACLES:
        # 曼哈顿距离
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    else:
        # 欧几里得距离
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def get_neighbors(node):
    """获取节点的相邻节点"""
    # 如果没有障碍物，只使用上下左右四个方向
    if not OBSTACLES:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else:
        # 有障碍物时使用8个方向
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    
    neighbors = []
    for dir_x, dir_y in directions:
        neighbor = (node[0] + dir_x, node[1] + dir_y)
        # 检查是否在地图范围内且不是障碍物
        if (0 <= neighbor[0] < MAP_SIZE[0] and 
            0 <= neighbor[1] < MAP_SIZE[1] and 
            neighbor not in OBSTACLES):
            neighbors.append(neighbor)
    
    return neighbors

def parse_coordinates(coord_str):
    """从坐标字符串解析坐标值"""
    try:
        # 处理格式如 "10,20" 或 "(10, 20)" 的坐标
        coord_str = coord_str.strip('() ')
        x, y = map(int, coord_str.split(','))
        # 确保坐标在地图范围内
        x = max(0, min(GRID_SIZE - 1, x))
        y = max(0, min(GRID_SIZE - 1, y))
        return (x, y)
    except:
        # 默认返回随机有效坐标
        return (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

def astar_algorithm(start_point_str, end_point_str):
    """A*路径规划算法"""
    # 解析起点和终点坐标
    start = parse_coordinates(start_point_str)
    goal = parse_coordinates(end_point_str)
    
    print(f"起点坐标: {start}")
    print(f"终点坐标: {goal}")
    
    # 如果没有障碍物，直接返回起点和终点构成的直线路径
    if not OBSTACLES:
        # 确保起点和终点都是整数坐标
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        # 计算路径上的所有点
        path = []
        x1, y1 = start
        x2, y2 = goal
        
        # 使用Bresenham算法生成直线上的所有点
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            path.append((x, y))
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        # 确保路径包含终点
        if path[-1] != goal:
            path.append(goal)
        
        total_distance = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        estimated_time = total_distance / 10
        
        path_data = {
            "points": [{"x": float(point[0]), "y": float(point[1])} for point in path],
            "obstacles": []
        }
        
        return path_data, float(total_distance), estimated_time
    
    # 有障碍物时使用完整的A*算法
    # 确保起点和终点不是障碍物
    if start in OBSTACLES:
        OBSTACLES.remove(start)
    if goal in OBSTACLES:
        OBSTACLES.remove(goal)
    
    # 初始化open_set
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # 记录从起点到每个节点的最短路径
    came_from = {}
    
    # 记录从起点到每个节点的成本
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    
    # 记录从起点经过每个节点到终点的估计成本
    f_score = defaultdict(lambda: float('inf'))
    f_score[start] = heuristic(start, goal)
    
    # 记录已访问的节点
    closed_set = set()
    
    while open_set:
        # 获取f_score最小的节点
        current_f, current = heapq.heappop(open_set)
        
        # 如果到达终点，构建路径并返回
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
            # 计算总距离
            total_distance = 0
            for i in range(len(path) - 1):
                total_distance += heuristic(path[i], path[i + 1])
            
            # 假设船的平均速度为10海里/小时
            estimated_time = total_distance / 10
            
            # 构建路径数据
            path_data = {
                "points": [{"x": float(point[0]), "y": float(point[1])} for point in path],
                "obstacles": [{"x": float(obs[0]), "y": float(obs[1])} for obs in list(OBSTACLES)[:100]]
            }
            
            return path_data, float(total_distance), estimated_time
        
        # 将当前节点加入closed_set
        closed_set.add(current)
        
        # 检查所有相邻节点
        for neighbor in get_neighbors(current):
            # 如果邻居节点已经在closed_set中，跳过
            if neighbor in closed_set:
                continue
            
            # 计算经过当前节点到达相邻节点的成本
            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            
            # 如果找到了更优的路径，则更新
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                # 将相邻节点加入open_set
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # 如果找不到路径，返回直线路径
    path = [start, goal]
    total_distance = heuristic(start, goal)
    estimated_time = total_distance / 10
    
    path_data = {
        "points": [{"x": float(point[0]), "y": float(point[1])} for point in path],
        "obstacles": [{"x": float(obs[0]), "y": float(obs[1])} for obs in list(OBSTACLES)[:100]]
    }
    
    return path_data, float(total_distance), estimated_time

# DDPG算法实现（深度确定性策略梯度）
# 环境参数
STATE_DIM = 4  # 状态维度：[当前x, 当前y, 目标x, 目标y]
ACTION_DIM = 2  # 动作维度：[x方向移动, y方向移动]
MAX_ACTION = 1.0  # 动作最大值

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 神经网络参数
HIDDEN_DIM = 256  # 隐藏层维度
LEARNING_RATE_ACTOR = 5e-5  # actor学习率
LEARNING_RATE_CRITIC = 1e-4  # critic学习率
GAMMA = 0.99  # 折扣因子
TAU = 0.001  # 软更新参数
BATCH_SIZE = 256  # 批量大小
BUFFER_SIZE = 100000  # 经验回放缓冲区大小
NOISE_STD = 0.2  # 噪声标准差
NOISE_DECAY = 0.9999  # 噪声衰减因子

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
        # 移动到GPU
        self.to(DEVICE)
        
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.fc3(x)) * MAX_ACTION

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
        # 移动到GPU
        self.to(DEVICE)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(state).to(DEVICE),
                torch.FloatTensor(action).to(DEVICE),
                torch.FloatTensor(reward).unsqueeze(1).to(DEVICE),
                torch.FloatTensor(next_state).to(DEVICE),
                torch.FloatTensor(done).unsqueeze(1).to(DEVICE))
    
    def __len__(self):
        return len(self.buffer)

class GaussianNoise:
    def __init__(self, action_dim, std=NOISE_STD):
        self.action_dim = action_dim
        self.std = std
        self.decay = NOISE_DECAY
        
    def sample(self):
        return np.random.normal(0, self.std, size=self.action_dim)
    
    def decay_noise(self):
        self.std *= self.decay

class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=LEARNING_RATE_ACTOR)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=LEARNING_RATE_CRITIC)
        
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.noise = GaussianNoise(action_dim)
    
    def select_action(self, state):
        with torch.no_grad():
            # 确保状态是在GPU上的tensor
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(DEVICE)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            action = self.actor(state)
            action = action.cpu().numpy().squeeze()
            
            # 添加噪声
            action = action + self.noise.sample()
            return np.clip(action, -MAX_ACTION, MAX_ACTION)
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return 0, 0
        
        # 从经验回放缓冲区采样
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        
        # 更新Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * GAMMA * target_Q
        
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()
    
    def decay_noise(self):
        self.noise.decay_noise()
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

def get_reward(current_state, next_state, goal, start_to_goal_dist):
    """
    计算奖励函数
    Args:
        current_state: 当前状态
        next_state: 下一状态
        goal: 目标位置
        start_to_goal_dist: 起点到终点的距离
    """
    # 计算距离奖励: 负的下一位置到终点的距离 / 起点到终点的距离
    next_to_goal_dist = math.sqrt((next_state[0] - goal[0])**2 + (next_state[1] - goal[1])**2)
    distance_reward = -next_to_goal_dist / start_to_goal_dist
    
    # 计算障碍物奖励
    obstacle_penalty = 0
    obstacle_threshold = 0.2
    min_obstacle_dist = float('inf')
    
    for obs in OBSTACLES:
        dist = math.sqrt((next_state[0] - obs[0])**2 + (next_state[1] - obs[1])**2)
        min_obstacle_dist = min(min_obstacle_dist, dist)
        
    if min_obstacle_dist < obstacle_threshold:
        obstacle_penalty = -10 * (obstacle_threshold - min_obstacle_dist) / obstacle_threshold
    
    # 计算边界奖励
    boundary_penalty = 0
    if (next_state[0] < 0 or next_state[0] >= MAP_SIZE[0] or 
        next_state[1] < 0 or next_state[1] >= MAP_SIZE[1]):
        boundary_penalty = -10  # 降低边界惩罚
    
    # 计算总奖励
    total_reward = distance_reward + obstacle_penalty + boundary_penalty
    
    # 到达目标的额外奖励
    if next_to_goal_dist < 1.5:
        total_reward += 100
        
    return total_reward

def plot_training_curves(rewards, critic_losses, actor_losses, save_path='training_curves.png'):
    """绘制训练曲线"""
    # 设置非交互式后端
    import matplotlib
    matplotlib.use('Agg')
    
    plt.figure(figsize=(15, 5))
    
    # 平滑处理
    window_size = 10
    
    # 绘制奖励曲线
    plt.subplot(1, 3, 1)
    if len(rewards) > window_size:
        rewards_smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(rewards_smoothed, label='Smoothed')
        plt.plot(rewards, alpha=0.3, label='Raw')
    else:
        plt.plot(rewards, label='Rewards')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # 绘制Critic损失曲线
    plt.subplot(1, 3, 2)
    if len(critic_losses) > window_size:
        critic_losses_smoothed = np.convolve(critic_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(critic_losses_smoothed, label='Smoothed')
        plt.plot(critic_losses, alpha=0.3, label='Raw')
    else:
        plt.plot(critic_losses, label='Critic Losses')
    plt.title('Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制Actor损失曲线
    plt.subplot(1, 3, 3)
    if len(actor_losses) > window_size:
        actor_losses_smoothed = np.convolve(actor_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(actor_losses_smoothed, label='Smoothed')
        plt.plot(actor_losses, alpha=0.3, label='Raw')
    else:
        plt.plot(actor_losses, label='Actor Losses')
    plt.title('Actor Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n训练曲线已保存到 {save_path}")

def ddpg_algorithm(start_point_str, end_point_str):
    """使用DDPG算法进行路径规划"""
    # 解析起点和终点坐标
    start = np.array(parse_coordinates(start_point_str), dtype=np.float32)
    goal = np.array(parse_coordinates(end_point_str), dtype=np.float32)
    
    print(f"起点坐标: {start}")
    print(f"终点坐标: {goal}")
    
    # 计算起点到终点的距离（用于奖励标准化）
    start_to_goal_dist = math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
    
    # 初始化DDPG智能体
    agent = DDPGAgent(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    
    # 训练参数
    max_episodes = 150
    max_steps = 200
    best_reward = float('-inf')
    best_path = None
    best_distance = 0
    
    # 用于记录训练指标
    episode_rewards = []
    episode_critic_losses = []
    episode_actor_losses = []
    
    # 训练循环
    for episode in range(max_episodes):
        state = np.array([*start, *goal])
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        train_steps = 0
        current_path = [tuple(start)]
        current_distance = 0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(range(max_steps), desc=f'Episode {episode + 1}/{max_episodes}')
        
        for step in progress_bar:
            # 选择动作
            action = agent.select_action(state)
            
            # 计算当前点到目标的距离
            current_to_goal_dist = math.sqrt((state[0] - goal[0])**2 + (state[1] - goal[1])**2)
            
            # 计算当前点到最近障碍物的距离
            min_obstacle_dist = float('inf')
            for obs in OBSTACLES:
                dist = math.sqrt((state[0] - obs[0])**2 + (state[1] - obs[1])**2)
                min_obstacle_dist = min(min_obstacle_dist, dist)
            
            # 根据距离动态调整步长
            step_scale = 1.0
            if current_to_goal_dist >= 10 and min_obstacle_dist >= 1:
                step_scale = 2.0
            
            # 执行动作（使用动态步长）
            next_x = state[0] + action[0] * step_scale
            next_y = state[1] + action[1] * step_scale
            
            # 创建下一状态
            next_state = np.array([next_x, next_y, goal[0], goal[1]])
            
            # 计算奖励
            reward = get_reward(state[:2], next_state[:2], goal, start_to_goal_dist)
            episode_reward += reward
            
            # 检查是否到达目标
            done = math.sqrt((next_state[0] - goal[0])**2 + (next_state[1] - goal[1])**2) < 1.5
            
            # 更新距离
            step_distance = math.sqrt((next_state[0] - state[0])**2 + (next_state[1] - state[1])**2)
            current_distance += step_distance
            
            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)
            
            # 训练智能体
            if len(agent.memory) > BATCH_SIZE:
                critic_loss, actor_loss = agent.train()
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                train_steps += 1
            
            # 更新状态和路径
            state = next_state
            
            # 如果智能体已经到达目标位置，则提前结束本轮
            if done:
                # 确保路径包含终点
                current_path.append(tuple(goal))
                break
                
            # 如果智能体出界，则将智能体拉回边界而不是终止训练
            if state[0] < 0:
                state[0] = 0
            elif state[0] >= MAP_SIZE[0]:
                state[0] = MAP_SIZE[0] - 1
            
            if state[1] < 0:
                state[1] = 0
            elif state[1] >= MAP_SIZE[1]:
                state[1] = MAP_SIZE[1] - 1
                
            # 添加到路径中
            current_path.append((float(state[0]), float(state[1])))
            
            # 更新进度条
            progress_bar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Distance': f'{current_distance:.2f}',
                'Goal Dist': f'{current_to_goal_dist:.2f}',
                'Steps': step + 1
            })
        
        # 如果路径没有到达目标点，强制添加目标点
        if len(current_path) > 0 and current_path[-1] != tuple(goal):
            current_path.append(tuple(goal))
            current_distance += math.sqrt((current_path[-2][0] - goal[0])**2 + 
                                        (current_path[-2][1] - goal[1])**2)
        
        # 记录训练指标
        episode_rewards.append(episode_reward)
        if train_steps > 0:
            episode_critic_losses.append(episode_critic_loss / train_steps)
            episode_actor_losses.append(episode_actor_loss / train_steps)
        else:
            episode_critic_losses.append(0)
            episode_actor_losses.append(0)
        
        # 更新最优路径
        if done or episode_reward > best_reward:
            best_reward = episode_reward
            best_path = current_path
            best_distance = current_distance
            agent.save('ddpg_model.pth')
            print(f'\nEpisode {episode+1}: 找到更好的路径! 奖励: {best_reward:.2f}, 距离: {best_distance:.2f}')
        
        # 衰减噪声
        agent.decay_noise()
        
        # 打印本轮统计信息
        print(f'Episode {episode+1}: Reward={episode_reward:.2f}, Steps={len(current_path)}, Path Length={current_distance:.2f}')
    
    # 绘制训练曲线
    plot_training_curves(
        episode_rewards,
        episode_critic_losses,
        episode_actor_losses,
        save_path='training_curves.png'
    )
    
    # 如果没有找到有效路径，使用最后一轮的路径
    if best_path is None:
        best_path = current_path
        best_distance = current_distance
    
    # 计算估计时间（假设船的平均速度为10海里/小时）
    estimated_time = float(best_distance) / 10
    
    # 构建路径数据
    path_data = {
        "points": [{"x": float(point[0]), "y": float(point[1])} for point in best_path],
        "obstacles": [{"x": float(obs[0]), "y": float(obs[1])} for obs in list(OBSTACLES)[:100]]
    }
    
    return path_data, float(best_distance), estimated_time 