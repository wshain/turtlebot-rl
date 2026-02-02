#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        
        # 奖励参数配置
        self.reward_config = {
            'collision_penalty': -500,        # 碰撞惩罚
            'goal_reward': 1000,             # 到达目标奖励
            'obstacle_penalty': -5,          # 障碍物惩罚
            'obstacle_safe_reward': 1,       # 安全距离奖励
            'min_obstacle_distance': 0.5,    # 最小安全距离
            'progress_reward_factor': 2,     # 进度奖励因子
            'efficiency_reward': 0.1,        # 效率奖励
            'time_penalty': -0.1,            # 时间惩罚（鼓励快速到达）
            'smoothness_reward': 0.05,       # 平滑运动奖励
            'distance_threshold': 0.2,       # 到达目标阈值
            'obstacle_gradient_range': 1.0   # 障碍物梯度范围
        }
        
        # 记录上一步状态用于平滑度计算
        self.last_action = None
        self.last_heading = None

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

    def setReward(self, state, done, action):
        """
        高级奖励函数，包含多种奖励机制
        """
        config = self.reward_config
        
        yaw_reward = []
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        # 1. 方向奖励计算
        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        # 2. 距离进度奖励
        distance_rate = config['progress_reward_factor'] ** (current_distance / self.goal_distance)

        # 3. 障碍物奖励 - 梯度设计
        if obstacle_min_range < config['min_obstacle_distance']:
            ob_reward = config['obstacle_penalty']
        elif obstacle_min_range < config['obstacle_gradient_range']:
            # 在安全距离范围内，奖励逐渐增加
            gradient_factor = (obstacle_min_range - config['min_obstacle_distance']) / (config['obstacle_gradient_range'] - config['min_obstacle_distance'])
            ob_reward = config['obstacle_safe_reward'] * gradient_factor
        else:
            ob_reward = config['obstacle_safe_reward']

        # 4. 效率奖励 - 鼓励朝向目标方向移动
        efficiency_reward = config['efficiency_reward'] * abs(yaw_reward[action])

        # 5. 平滑运动奖励 - 减少频繁转向
        smoothness_reward = 0
        if self.last_action is not None:
            action_diff = abs(action - self.last_action)
            if action_diff <= 1:  # 连续或相邻动作
                smoothness_reward = config['smoothness_reward']
            elif action_diff >= 3:  # 大幅转向
                smoothness_reward = -config['smoothness_reward']

        # 6. 时间惩罚 - 鼓励快速到达
        time_penalty = config['time_penalty']

        # 7. 距离接近奖励 - 当接近目标时给予额外奖励
        proximity_reward = 0
        if current_distance < 0.5:  # 接近目标时
            proximity_reward = (0.5 - current_distance) * 10

        # 总奖励计算
        reward = (
            (round(yaw_reward[action] * 5, 2)) * distance_rate +  # 方向奖励
            ob_reward +                                           # 障碍物奖励
            efficiency_reward +                                   # 效率奖励
            smoothness_reward +                                   # 平滑运动奖励
            time_penalty +                                        # 时间惩罚
            proximity_reward                                      # 接近奖励
        )

        # 碰撞处理
        if done:
            rospy.loginfo("Collision!!")
            reward = config['collision_penalty']
            self.pub_cmd_vel.publish(Twist())

        # 到达目标处理
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = config['goal_reward']
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        # 更新历史状态
        self.last_action = action
        self.last_heading = heading

        return reward

    def updateRewardConfig(self, new_config):
        """
        动态更新奖励配置
        """
        self.reward_config.update(new_config)
        rospy.loginfo("Reward configuration updated: %s", new_config)

    def getRewardConfig(self):
        """
        获取当前奖励配置
        """
        return self.reward_config.copy()


    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)