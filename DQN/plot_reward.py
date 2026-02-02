import numpy as np
import matplotlib.pyplot as plt
import os

# 检查文件是否存在
files = ['rewards_log.npy', 'losses_log.npy', 'steps_log.npy']
for f in files:
    if not os.path.exists(f):
        print(f"File {f} not found. Please run the training first.")
        exit()

# 加载数据
rewards = np.load('rewards_log.npy')
losses = np.load('losses_log.npy')
steps = np.load('steps_log.npy')

# 平滑处理（可选）
def smooth(x, window_len=50):
    if len(x) < window_len:
        return x
    return np.convolve(x, np.ones(window_len)/window_len, mode='valid')

# 绘图
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(smooth(rewards), color='tab:blue')
plt.title('Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(smooth(losses), color='tab:orange')
plt.title('Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(smooth(steps), color='tab:green')
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=300)
plt.show()