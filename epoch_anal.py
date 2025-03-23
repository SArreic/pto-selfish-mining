import os
import re
import matplotlib.pyplot as plt

# 手动设置 epoch 长度
length = 3000  # 每 length 次动作计算一次

# 文件名
filename = "output.txt"

# 正则匹配 'actions' 部分
reward_pattern = re.compile(r"'actions':\s*{([^}]+)}")

# 统计数据容器
epoch_rewards = []  # 存储每个 epoch 的最高奖励总和的平均值
current_epoch_rewards = []  # 当前 epoch 内的最高奖励集合


def extract_max_reward(line):
    """ 从日志行中提取最大奖励 """
    match = reward_pattern.search(line)
    if match:
        actions_str = match.group(1)
        actions = eval("{" + actions_str + "}")
        return max(actions.values()) if actions else 0.0
    return None


# 读取文件并解析
with open(filename, "r", encoding="utf-16") as file:
    for idx, line in enumerate(file):
        max_reward = extract_max_reward(line)
        if max_reward is not None:
            current_epoch_rewards.append(max_reward)

        # 每 length 次记录一次最高奖励之和的平均值
        if (idx + 1) % length == 0 and current_epoch_rewards:
            avg_max_reward = sum(current_epoch_rewards) / len(current_epoch_rewards)
            epoch_rewards.append(avg_max_reward)
            current_epoch_rewards = []  # 重置当前 epoch 记录

# 生成折线图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(epoch_rewards) + 1), epoch_rewards, marker='o', linestyle='-', color='b')
plt.xlabel("Epoch ({} steps each)".format(length))
plt.ylabel("Average Max Reward per Epoch")
plt.title("Average Max Reward Over Epochs")
plt.grid(True)
plt.tight_layout()

# 保存图像
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "epoch_max_reward.png"))
plt.show()

print("分析完成，结果已保存到 'analysis_results/epoch_max_reward.png'")
