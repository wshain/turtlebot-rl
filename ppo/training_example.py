#!/usr/bin/env python
"""
训练示例文件
展示如何使用新的奖励配置系统进行训练
"""

import rospy
from environment_stage_4 import Env
from reward_config_examples import get_config_by_name

def train_with_dynamic_rewards():
    """
    使用动态奖励配置进行训练
    """
    # 初始化环境
    env = Env(action_size=5)
    
    # 训练阶段配置
    training_phases = [
        ('learning', 1000),      # 学习阶段：1000步
        ('basic', 2000),         # 基础阶段：2000步
        ('conservative', 1500),  # 保守阶段：1500步
        ('aggressive', 1500),    # 激进阶段：1500步
        ('fine_tuning', 1000)    # 精细调优：1000步
    ]
    
    total_steps = 0
    
    for phase_name, steps in training_phases:
        print(f"\n开始 {phase_name} 阶段训练，步数: {steps}")
        
        # 更新奖励配置
        config = get_config_by_name(phase_name)
        env.updateRewardConfig(config)
        
        # 在这个阶段进行训练
        for step in range(steps):
            # 这里应该是您的训练逻辑
            # 例如：action = agent.choose_action(state)
            #      next_state, reward, done = env.step(action)
            
            # 示例：随机动作
            import random
            action = random.randint(0, 4)
            
            # 执行动作
            state, reward, done = env.step(action)
            
            # 记录奖励
            if step % 100 == 0:
                print(f"步骤 {total_steps + step}: 奖励 = {reward:.2f}")
            
            if done:
                state = env.reset()
        
        total_steps += steps
        print(f"{phase_name} 阶段完成，总步数: {total_steps}")

def adaptive_reward_training():
    """
    自适应奖励训练 - 根据性能动态调整奖励
    """
    env = Env(action_size=5)
    
    # 初始配置
    current_config = get_config_by_name('basic')
    env.updateRewardConfig(current_config)
    
    # 性能跟踪
    episode_rewards = []
    collision_rate = 0
    success_rate = 0
    
    for episode in range(100):
        episode_reward = 0
        collisions = 0
        successes = 0
        
        state = env.reset()
        done = False
        
        while not done:
            # 训练逻辑
            action = random.randint(0, 4)  # 示例：随机动作
            state, reward, done = env.step(action)
            episode_reward += reward
            
            # 检测碰撞和成功
            if reward == current_config['collision_penalty']:
                collisions += 1
            elif reward == current_config['goal_reward']:
                successes += 1
        
        episode_rewards.append(episode_reward)
        
        # 每10个episode调整一次奖励
        if episode % 10 == 0 and episode > 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            collision_rate = collisions / 10
            success_rate = successes / 10
            
            print(f"Episode {episode}: 平均奖励={avg_reward:.2f}, 碰撞率={collision_rate:.2f}, 成功率={success_rate:.2f}")
            
            # 根据性能调整奖励
            if collision_rate > 0.3:  # 碰撞率过高
                print("碰撞率过高，切换到保守配置")
                new_config = get_config_by_name('conservative')
                env.updateRewardConfig(new_config)
                current_config = new_config
            elif success_rate < 0.2:  # 成功率过低
                print("成功率过低，切换到学习配置")
                new_config = get_config_by_name('learning')
                env.updateRewardConfig(new_config)
                current_config = new_config
            elif avg_reward > 500:  # 性能良好
                print("性能良好，切换到激进配置")
                new_config = get_config_by_name('aggressive')
                env.updateRewardConfig(new_config)
                current_config = new_config

def custom_reward_experiment():
    """
    自定义奖励实验
    """
    env = Env(action_size=5)
    
    # 自定义配置
    custom_config = {
        'collision_penalty': -800,        # 更严重的碰撞惩罚
        'goal_reward': 1500,             # 更高的目标奖励
        'obstacle_penalty': -8,          # 更严重的障碍物惩罚
        'obstacle_safe_reward': 1.5,     # 更高的安全奖励
        'min_obstacle_distance': 0.6,    # 更大的安全距离
        'progress_reward_factor': 2.5,   # 更高的进度奖励
        'efficiency_reward': 0.15,       # 更高的效率奖励
        'time_penalty': -0.15,           # 更大的时间压力
        'smoothness_reward': 0.08,       # 更高的平滑运动奖励
        'distance_threshold': 0.2,
        'obstacle_gradient_range': 1.2
    }
    
    env.updateRewardConfig(custom_config)
    print("使用自定义奖励配置进行训练...")
    
    # 训练逻辑
    for episode in range(50):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = random.randint(0, 4)
            state, reward, done = env.step(action)
            episode_reward += reward
        
        if episode % 10 == 0:
            print(f"Episode {episode}: 总奖励 = {episode_reward:.2f}")

if __name__ == "__main__":
    try:
        rospy.init_node('reward_training_example')
        
        print("=== 奖励配置训练示例 ===")
        print("1. 动态奖励训练")
        print("2. 自适应奖励训练")
        print("3. 自定义奖励实验")
        
        choice = input("请选择训练模式 (1-3): ")
        
        if choice == '1':
            train_with_dynamic_rewards()
        elif choice == '2':
            adaptive_reward_training()
        elif choice == '3':
            custom_reward_experiment()
        else:
            print("无效选择，使用默认的动态奖励训练")
            train_with_dynamic_rewards()
            
    except rospy.ROSInterruptException:
        print("训练被中断")
    except KeyboardInterrupt:
        print("用户中断训练") 