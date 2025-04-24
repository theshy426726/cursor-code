import numpy as np
import math
import heapq
import random
import json
from collections import defaultdict

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
    """计算两点之间的欧几里得距离"""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def get_neighbors(node):
    """获取节点的相邻节点"""
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
        return (x, y)
    except:
        # 默认返回随机有效坐标
        return (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

def astar_algorithm(start_point_str, end_point_str):
    """A*路径规划算法"""
    # 解析起点和终点坐标
    start = parse_coordinates(start_point_str)
    goal = parse_coordinates(end_point_str)
    
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
                "points": [{"x": point[0], "y": point[1]} for point in path],
                "obstacles": [{"x": obs[0], "y": obs[1]} for obs in list(OBSTACLES)[:100]]  # 限制障碍物数量
            }
            
            return path_data, total_distance, estimated_time
        
        # 检查所有相邻节点
        for neighbor in get_neighbors(current):
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
        "points": [{"x": point[0], "y": point[1]} for point in path],
        "obstacles": [{"x": obs[0], "y": obs[1]} for obs in list(OBSTACLES)[:100]]
    }
    
    return path_data, total_distance, estimated_time

# DDPG算法实现（深度确定性策略梯度）
# 在实际应用中，应该使用深度学习框架如TensorFlow或PyTorch实现
# 这里为简化起见，提供一个模拟实现
def ddpg_algorithm(start_point_str, end_point_str):
    """DDPG路径规划算法（模拟实现）"""
    # 解析起点和终点坐标
    start = parse_coordinates(start_point_str)
    goal = parse_coordinates(end_point_str)
    
    # 确保起点和终点不是障碍物
    if start in OBSTACLES:
        OBSTACLES.remove(start)
    if goal in OBSTACLES:
        OBSTACLES.remove(goal)
    
    # 模拟DDPG生成的路径
    # 在实际实现中，这应该是通过神经网络生成的
    path = [start]
    current = start
    
    # 生成一条平滑的路径
    while heuristic(current, goal) > 2:
        # 计算朝向目标的方向
        direction = (
            (goal[0] - current[0]) / max(1, heuristic(current, goal)),
            (goal[1] - current[1]) / max(1, heuristic(current, goal))
        )
        
        # 添加一些随机扰动，模拟DDPG的探索特性
        noise_x = random.uniform(-0.5, 0.5)
        noise_y = random.uniform(-0.5, 0.5)
        
        # 计算下一个点
        next_x = int(current[0] + direction[0] * 2 + noise_x)
        next_y = int(current[1] + direction[1] * 2 + noise_y)
        
        # 确保在地图范围内
        next_x = max(0, min(MAP_SIZE[0] - 1, next_x))
        next_y = max(0, min(MAP_SIZE[1] - 1, next_y))
        
        # 避开障碍物
        if (next_x, next_y) in OBSTACLES:
            # 尝试找到附近的非障碍点
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    candidate = (next_x + dx, next_y + dy)
                    if (0 <= candidate[0] < MAP_SIZE[0] and 
                        0 <= candidate[1] < MAP_SIZE[1] and 
                        candidate not in OBSTACLES):
                        next_x, next_y = candidate
                        break
        
        current = (next_x, next_y)
        path.append(current)
    
    # 添加终点
    path.append(goal)
    
    # 计算总距离
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += heuristic(path[i], path[i + 1])
    
    # 假设船的平均速度为10海里/小时
    estimated_time = total_distance / 10
    
    # 构建路径数据
    path_data = {
        "points": [{"x": point[0], "y": point[1]} for point in path],
        "obstacles": [{"x": obs[0], "y": obs[1]} for obs in list(OBSTACLES)[:100]]  # 限制障碍物数量
    }
    
    return path_data, total_distance, estimated_time 