import os
import re
import matplotlib.pyplot as plt

# 每个 epoch 包含的步数
length = 3000

# 待处理的策略与文件名对应关系
strategies = {
    "greedy": "output_greedy_v2.txt",
    "random": "output_random.txt",
    "mcts": "output_mcts.txt"
}

# 正则表达式匹配 actions 部分
reward_pattern = re.compile(r"'actions':\s*{([^}]+)}")

# 结果保存目录
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)


def extract_max_reward(line):
    """ 从日志行中提取最大奖励 """
    match = reward_pattern.search(line)
    if match:
        actions_str = match.group(1)
        try:
            actions = eval("{" + actions_str + "}")
            return max(actions.values()) if actions else 0.0
        except:
            return 0.0
    return None


def process_file(filepath):
    """ 处理单个日志文件，返回每个 epoch 的平均最大奖励 """
    epoch_rewards = []
    current_epoch_rewards = []

    with open(filepath, "r", encoding="utf-16") as file:
        for idx, line in enumerate(file):
            max_reward = extract_max_reward(line)
            if max_reward is not None:
                current_epoch_rewards.append(max_reward)

            if (idx + 1) % length == 0 and current_epoch_rewards:
                avg_max_reward = sum(current_epoch_rewards) / len(current_epoch_rewards)
                epoch_rewards.append(avg_max_reward)
                current_epoch_rewards = []

    return epoch_rewards


# 存储所有策略对应的epoch奖励数据
all_results = {}

# 处理每个策略的数据
for strategy, filename in strategies.items():
    filepath = filename
    rewards = process_file(filepath)
    all_results[strategy] = rewards

    # 绘制该策略的单独图像
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', linestyle='-', label=strategy, color='b')
    plt.xlabel(f"Epoch ({length} steps each)")
    plt.ylabel("Average Max Reward per Epoch")
    plt.title(f"{strategy.capitalize()} Strategy - Average Max Reward Over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{strategy}_epoch_max_reward.png"))
    plt.close()


# 绘制所有策略两两对比图
def plot_comparison(strategy1, strategy2, rewards1, rewards2, color1, color2):
    """ 绘制两个策略的对比图 """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(rewards1) + 1), rewards1, marker='o', linestyle='-',
             label=f"{strategy1.capitalize()} Strategy", color=color1)
    plt.plot(range(1, len(rewards2) + 1), rewards2, marker='o', linestyle='-',
             label=f"{strategy2.capitalize()} Strategy", color=color2)
    plt.xlabel(f"Epoch ({length} steps each)")
    plt.ylabel("Average Max Reward per Epoch")
    plt.title(f"Comparison of {strategy1.capitalize()} vs {strategy2.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{strategy1}_{strategy2}_comparison_epoch_max_reward.png"))
    plt.close()


# 对每一对策略进行对比绘图并保存
plot_comparison("mcts", "greedy", all_results["mcts"], all_results["greedy"], "red", "green")
plot_comparison("mcts", "random", all_results["mcts"], all_results["random"], "red", "blue")
plot_comparison("greedy", "random", all_results["greedy"], all_results["random"], "green", "blue")

# 绘制所有策略对比图
plt.figure(figsize=(12, 6))
colors = {"random": "blue", "greedy": "green", "mcts": "red"}

for strategy, rewards in all_results.items():
    plt.plot(range(1, len(rewards) + 1), rewards,
             marker='o', linestyle='-', label=strategy.capitalize(), color=colors.get(strategy, 'black'))

plt.xlabel(f"Epoch ({length} steps each)")
plt.ylabel("Average Max Reward per Epoch")
plt.title("Comparison of Average Max Reward Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparison_epoch_max_reward.png"))
plt.show()

print("所有分析完成，图像已保存到 'analysis_results/' 文件夹中。")
