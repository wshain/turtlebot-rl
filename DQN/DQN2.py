import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, random
from environment_stage_4 import Env

import time
import rospy
from std_msgs.msg import Float32MultiArray
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# -----------------------------
# 超参数定义
# -----------------------------
BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.9
TARGET_REPLACE_ITER = 10
MEMORY_CAPACITY = 2000
N_ACTIONS = 5
N_STATES = 28

# -----------------------------
# 初始化环境
# -----------------------------
env = Env(N_ACTIONS)

# -----------------------------
# 神经网络定义
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 注意：这里不要对输出加 dropout！Q 值需要稳定
        return self.out(x)

# -----------------------------
# DQN 定义
# -----------------------------
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 更清晰的维度
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.start_epoch = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.load_models = True
        self.load_ep = 100
        self.loss = 0.0
        self.q_eval = 0.0
        self.q_target = 0.0

        if self.load_models:
            checkpoint = torch.load(f"./model/{self.load_ep}.pt")
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.epsilon = 0.0  # 加载模型后 epsilon 可以从 0 开始，或继续衰减
            print("✅ Model loaded successfully.")

    def choose_action(self, x):
        x = torch.FloatTensor(x).unsqueeze(0)
        if np.random.uniform() > self.epsilon:
            actions_value = self.eval_net(x)
            action = actions_value.max(1)[1].item()
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(min(self.memory_counter, MEMORY_CAPACITY), BATCH_SIZE)
        b_memory = self.memory[sample_index]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.loss = loss.item()
        self.q_eval = q_eval.mean().item()
        self.q_target = q_target.mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, name):
        state = {
            'target_net': self.target_net.state_dict(),
            'eval_net': self.eval_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.start_epoch
        }
        torch.save(state, f"./model/{name}.pt")
        print(f"💾 Model saved: ./model/{name}.pt")

# -----------------------------
# ✅ 数据记录列表（训练中保存）
# -----------------------------
rewards_log = []
losses_log = []
steps_log = []

# -----------------------------
# 主程序
# -----------------------------
if __name__ == '__main__':
    dqn = DQN()
    rospy.init_node('turtlebot3_dqn_stage_4')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    start_time = time.time()
    e = dqn.start_epoch

    # ✅ 创建 model 目录（避免报错）
    os.makedirs("./model", exist_ok=True)

    for e in range(e, e + 10000):
        s = env.reset()
        rospy.loginfo(f"🎯 New goal at ({env.goal_x:.2f}, {env.goal_y:.2f})")
        episode_reward = 0
        done = False
        max_steps = 2500  # 防止卡住！

        for t in range(max_steps):
            a = dqn.choose_action(s)
            s_, r, done = env.step(a)
            dqn.store_transition(s, a, r, s_)
            episode_reward += r
            s = s_

            # 学习
            if dqn.memory_counter > BATCH_SIZE:
                dqn.learn()

            # ✅ 每100步保存一次模型（可选）
            if t % 100 == 0 and e % 10 == 0:
                dqn.save_model(str(e))

            # ✅ 超时判断（防止卡住）
            if t >= 2499:
                rospy.logwarn("⏰ Episode %d timed out at step %d", e, t)
                done = True

            # 结束 episode
            if done:
                break

            # epsilon 衰减
            if dqn.epsilon > dqn.epsilon_min:
                dqn.epsilon -= 0.0001

        # --- ✅ 记录数据 ---
        rewards_log.append(episode_reward)
        losses_log.append(dqn.loss)
        steps_log.append(t)

        # --- 发布结果 ---
        result.data = [episode_reward, dqn.loss, dqn.q_eval, dqn.q_target]
        pub_result.publish(result)

        # --- 日志 ---
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo('Ep: %d | Score: %.2f | Steps: %d | Mem: %d | Eps: %.3f | Time: %d:%02d:%02d',
                      e, episode_reward, t, dqn.memory_counter, dqn.epsilon, h, m, s)

        # --- ✅ 每100轮保存一次数据 ---
        if e % 100 == 0:
            np.save("rewards_log.npy", np.array(rewards_log))
            np.save("losses_log.npy", np.array(losses_log))
            np.save("steps_log.npy", np.array(steps_log))
            dqn.save_model(str(e))
            rospy.loginfo("📊 Data and model saved at episode %d", e)

    # --- ✅ 训练结束，保存全部 ---
    np.save("rewards_log.npy", np.array(rewards_log))
    np.save("losses_log.npy", np.array(losses_log))
    np.save("steps_log.npy", np.array(steps_log))
    rospy.loginfo("🎉 Training finished. All logs saved.")

# import torch                                    # 导入torch
# import torch.nn as nn                           # 导入torch.nn
# import torch.nn.functional as F                 # 导入torch.nn.functional
# import numpy as np                              # 导入numpy
# # import gym                                      # 导入gym
# import math, random
# from environment_stage_4 import Env

# import time
# import rospy
# from std_msgs.msg import Float32MultiArray
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# # 超参数
# BATCH_SIZE = 128                               # 样本数量
# LR = 0.001                                       # 学习率
# EPISODES=0.99                          # greedy policy
# GAMMA = 0.9                                     # reward discount
# TARGET_REPLACE_ITER = 10                       # 目标网络更新频率
# MEMORY_CAPACITY = 2000                          # 记忆库容量
# N_ACTIONS = 5
# env =Env(N_ACTIONS)        # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)             # 杆子动作个数 (2个)
# N_STATES =  28  # 杆子状态个数 (4个)


# """
# torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可以用来定义和运行神经网络。
# nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。
# 定义网络：
#     需要继承nn.Module类，并实现forward方法。
#     一般把网络中具有可学习参数的层放在构造函数__init__()中。
#     只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
# """


# # 定义Net类 (定义网络)
# class Net(nn.Module):
#     def __init__(self):                                                         # 定义Net的一系列属性
#         # nn.Module的子类函数必须在构造函数中执行父类的构造函数
#         super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

#         self.fc1 = nn.Linear(N_STATES, 128)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
#         self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
#         self.fc2 =nn.Linear(128,128)
#         self.fc2.weight.data.normal_(0,0.1)
#         self.out = nn.Linear(128, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
#         self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

#     def forward(self, x):                                                       # 定义forward函数 (x为状态)
#         x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
#         x=F.relu(self.fc2(x))
#         x=F.dropout(self.fc2(x))
#         actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
#         return actions_value                                                    # 返回动作值


# # 定义DQN类 (定义两个网络)
# class DQN(object):
#     def __init__(self):                                                         # 定义DQN的一系列属性
#         self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
#         self.learn_step_counter = 0                                             # for target updating
#         self.memory_counter = 0                                                 # for storing memory
#         self.memory = np.zeros((MEMORY_CAPACITY, 58))             # 初始化记忆库，一行代表一个transition
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
#         self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
#         self.start_epoch = 0
#         self.epsilon = 1.0
#         self.epsilon_min = 0.01
#         self.load_models=True
#         self.load_ep =200
#         self.loss =0
#         self.q_eval=0
#         self.q_target=0
#         if self.load_models:
#             self.epsilon= 0
#             self.start_epoch=self.load_ep
#             checkpoint = torch.load("./model/"+str(self.load_ep)+".pt")
#             print(checkpoint.keys())
#             print(checkpoint['epoch'])
#             print(checkpoint)
#             self.target_net.load_state_dict(checkpoint['target_net'])
#             self.eval_net.load_state_dict(checkpoint['eval_net'])
#             self.optimizer.load_state_dict(checkpoint['optimizer'])
#             self.start_epoch = checkpoint['epoch'] + 1
#             print("loadmodel")
#     def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
#         x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
#         if np.random.uniform() > self.epsilon:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
#             actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
#             action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式

#             action = action[0]                                                  # 输出action的第一个数
#         else:                                                                   # 随机选择动作
#             action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
#         return action                                                           # 返回选择的动作 (0或1)

#     def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
#         transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
#         # 如果记忆库满了，便覆盖旧的数据
#         index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
#         self.memory[index, :] = transition                                      # 置入transition
#         self.memory_counter += 1                                                # memory_counter自加1

#     def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
#         # 目标网络参数更新
#         if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
#             self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
#         self.learn_step_counter += 1                                            # 学习步数自加1

#         # 抽取记忆库中的批数据
#         sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
#         b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
#         b_s = torch.FloatTensor(b_memory[:, :N_STATES])
#         # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
#         b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
#         # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
#         b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
#         # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
#         b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
#         # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

#         # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
#         q_eval = self.eval_net(b_s).gather(1, b_a)
#         # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
#         q_next = self.target_net(b_s_).detach()
#         # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
#         q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
#         # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
#         loss = self.loss_func(q_eval, q_target)
#         self.loss = torch.max(loss)
#         self.q_eval =torch.max(q_eval)
#         self.q_target =torch.max(q_target)
#         # 输入32个评估值和32个目标值，使用均方损失函数
#         self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
#         loss.backward()                                                 # 误差反向传播, 计算参数更新值
#         self.optimizer.step()                                           # 更新评估网络的所有参数



        

#     def save_model(self,dir):
#         state = {'target_net':self.target_net.state_dict(),'eval_net':self.eval_net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':e}
#         torch.save(state,"./model/"+ dir+".pt")

# if __name__=='__main__':

#     dqn = DQN()                                                             # 令dqn=DQN类
#     rospy.init_node('turtlebot3_dqn_stage_4')
#     pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
#     pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
#     result = Float32MultiArray()
#     get_action = Float32MultiArray()
#     start_time =time.time()
#     e=dqn.start_epoch
# for e in range(10000):                                                    # 400个episode循环
#     s = env.reset()
#     rospy.loginfo(f"New goal at ({env.goal_x}, {env.goal_y})")                                                   # 重置环境
#     episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
#     done=False
#     episode_step=6000

#     for t in range(episode_step):                                                         # 开始一个episode (每一个循环代表一步)
#         a = dqn.choose_action(s)                                        # 输入该步对应的状态s，选择动作
#         s_, r, done = env.step(a)                                 # 执行动作，获得反馈

#         # # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
#         # x, x_dot, theta, theta_dot = s_
#         # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         # new_r = r1 + r2

#         dqn.store_transition(s, a, r, s_)                 # 存储样本
#         episode_reward_sum += r                           # 逐步加上一个episode内每个step的reward
#         s = s_                                                # 更新状态
#         pub_get_action.publish(get_action)
#         if dqn.memory_counter > BATCH_SIZE:              # 如果累计的transition数量超过了记忆库的固定容量2000
#             # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
#             dqn.learn()
#         if e % 10 ==0:
#             dqn.save_model(str(e))
#         if t >=2500:
#             rospy.loginfo("time out!")
#             done =True


#         if done:       # 如果done为True
#             # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
#             result.data =[episode_reward_sum,float(dqn.loss),float(dqn.q_eval),float(dqn.q_target)]
#             pub_result.publish(result)
#             m,s =divmod(int(time.time()- start_time),60)
#             h,m =divmod(m,60)
#             rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',e, episode_reward_sum, dqn.memory_counter, dqn.epsilon, h, m, s)
#             # print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
#             param_keys = ['epsilon']
#             param_values = [dqn.epsilon]
#             param_dictionary = dict(zip(param_keys, param_values))

#             break                                             # 该episode结束
#         if dqn.epsilon > dqn.epsilon_min :
#             dqn.epsilon =dqn.epsilon-0.0001