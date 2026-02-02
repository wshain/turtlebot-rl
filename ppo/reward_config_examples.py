#!/usr/bin/env python
"""
奖励配置示例文件
包含不同训练场景下的奖励参数配置
"""

# 基础配置 - 平衡型
BASIC_CONFIG = {
    'collision_penalty': -500,
    'goal_reward': 1000,
    'obstacle_penalty': -5,
    'obstacle_safe_reward': 1,
    'min_obstacle_distance': 0.5,
    'progress_reward_factor': 2,
    'efficiency_reward': 0.1,
    'time_penalty': -0.1,
    'smoothness_reward': 0.05,
    'distance_threshold': 0.2,
    'obstacle_gradient_range': 1.0
}

# 保守配置 - 更注重安全性
CONSERVATIVE_CONFIG = {
    'collision_penalty': -1000,      # 更严重的碰撞惩罚
    'goal_reward': 800,
    'obstacle_penalty': -10,         # 更严重的障碍物惩罚
    'obstacle_safe_reward': 2,       # 更高的安全奖励
    'min_obstacle_distance': 0.8,    # 更大的安全距离
    'progress_reward_factor': 1.5,   # 降低进度奖励
    'efficiency_reward': 0.05,
    'time_penalty': -0.05,           # 减少时间压力
    'smoothness_reward': 0.1,        # 更注重平滑运动
    'distance_threshold': 0.2,
    'obstacle_gradient_range': 1.5   # 更大的梯度范围
}

# 激进配置 - 更注重效率
AGGRESSIVE_CONFIG = {
    'collision_penalty': -300,       # 较轻的碰撞惩罚
    'goal_reward': 1200,            # 更高的目标奖励
    'obstacle_penalty': -3,          # 较轻的障碍物惩罚
    'obstacle_safe_reward': 0.5,
    'min_obstacle_distance': 0.3,    # 更小的安全距离
    'progress_reward_factor': 3,     # 更高的进度奖励
    'efficiency_reward': 0.2,        # 更高的效率奖励
    'time_penalty': -0.2,            # 更大的时间压力
    'smoothness_reward': 0.02,       # 不太注重平滑运动
    'distance_threshold': 0.2,
    'obstacle_gradient_range': 0.8
}

# 学习阶段配置 - 适合初期训练
LEARNING_CONFIG = {
    'collision_penalty': -200,       # 较轻的惩罚，鼓励探索
    'goal_reward': 500,             # 适中的目标奖励
    'obstacle_penalty': -2,          # 较轻的障碍物惩罚
    'obstacle_safe_reward': 0.5,
    'min_obstacle_distance': 0.4,
    'progress_reward_factor': 1.8,   # 适中的进度奖励
    'efficiency_reward': 0.08,
    'time_penalty': -0.05,           # 较小的时间压力
    'smoothness_reward': 0.03,
    'distance_threshold': 0.2,
    'obstacle_gradient_range': 1.2
}

# 精细调优配置 - 适合后期优化
FINE_TUNING_CONFIG = {
    'collision_penalty': -600,
    'goal_reward': 1000,
    'obstacle_penalty': -6,
    'obstacle_safe_reward': 1.2,
    'min_obstacle_distance': 0.5,
    'progress_reward_factor': 2.2,
    'efficiency_reward': 0.12,
    'time_penalty': -0.12,
    'smoothness_reward': 0.06,
    'distance_threshold': 0.2,
    'obstacle_gradient_range': 1.0
}

def get_config_by_name(config_name):
    """
    根据配置名称获取奖励配置
    """
    configs = {
        'basic': BASIC_CONFIG,
        'conservative': CONSERVATIVE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG,
        'learning': LEARNING_CONFIG,
        'fine_tuning': FINE_TUNING_CONFIG
    }
    return configs.get(config_name, BASIC_CONFIG)

def print_config_summary(config_name, config):
    """
    打印配置摘要
    """
    print(f"\n=== {config_name.upper()} 配置 ===")
    print(f"碰撞惩罚: {config['collision_penalty']}")
    print(f"目标奖励: {config['goal_reward']}")
    print(f"障碍物惩罚: {config['obstacle_penalty']}")
    print(f"安全距离: {config['min_obstacle_distance']}m")
    print(f"进度奖励因子: {config['progress_reward_factor']}")
    print(f"时间惩罚: {config['time_penalty']}")

if __name__ == "__main__":
    # 打印所有配置摘要
    configs = {
        'basic': BASIC_CONFIG,
        'conservative': CONSERVATIVE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG,
        'learning': LEARNING_CONFIG,
        'fine_tuning': FINE_TUNING_CONFIG
    }
    
    for name, config in configs.items():
        print_config_summary(name, config) 