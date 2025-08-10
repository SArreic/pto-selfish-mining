import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# 每个 epoch 包含的步数
length = 3000

# 策略与文件名对应关系
strategies = {
    "ppo": "output_ppo.txt",
    "greedy": "output_greedy.txt",
    "random": "output_random.txt",
    # "mcts": "output_mcts.txt"
}

# 动作名称映射
action_names = {
    0: "Illegal",
    1: "Withhold",
    2: "Release",
    3: "Equivocate",
    4: "Vote"
}

# 日志匹配正则
decision_pattern = re.compile(r"\[(\w+) Decision\].*Action: (\d+), Estimated Value: ([\d\.\-eE]+)")
reward_pattern = re.compile(r"\[Env Feedback (\w+)\].*Action: (\d+), Reward: ([\d\.\-eE]+)")

# 输出目录
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)


def process_file(filepath):
    """ 处理单个日志文件，返回每个 epoch 的统计结果 """
    epoch_data = []
    current_epoch = defaultdict(lambda: {"rewards": [], "values": []})
    step_counter = 0

    with open(filepath, "r", encoding="utf-16") as file:
        for line in file:
            # 匹配决策
            decision_match = decision_pattern.search(line)
            if decision_match:
                strategy, action_str, value_str = decision_match.groups()
                action = int(action_str)
                value = float(value_str)
                current_epoch[action]["values"].append(value)
                step_counter += 1

            # 匹配奖励
            reward_match = reward_pattern.search(line)
            if reward_match:
                _, action_str, reward_str = reward_match.groups()
                action = int(action_str)
                reward = float(reward_str)
                current_epoch[action]["rewards"].append(reward)

            # 满一个 epoch，存储并重置
            if step_counter >= length:
                epoch_data.append(current_epoch)
                current_epoch = defaultdict(lambda: {"rewards": [], "values": []})
                step_counter = 0

    if current_epoch:
        epoch_data.append(current_epoch)

    return epoch_data


def compute_epoch_stats(epoch_data):
    """ 对每个 epoch 生成按动作分类的平均 reward 和 value """
    epoch_stats = []

    for epoch_idx, action_dict in enumerate(epoch_data):
        stats = {"epoch": epoch_idx + 1}
        for action in range(5):  # 0~4 动作
            name = action_names[action]
            rewards = action_dict[action]["rewards"]
            values = action_dict[action]["values"]

            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            avg_value = sum(values) / len(values) if values else 0.0

            stats[f"{name}_reward"] = avg_reward
            stats[f"{name}_value"] = avg_value
        epoch_stats.append(stats)

    return epoch_stats


def plot_action_metrics(stats, strategy, metric_type):
    """ 绘制某一策略中，不同动作的平均 reward 或 estimated value 随 epoch 变化 """
    plt.figure(figsize=(12, 6))
    for action in range(5):
        name = action_names[action]
        values = [epoch[f"{name}_{metric_type}"] for epoch in stats]
        plt.plot(range(1, len(values) + 1), values, label=name)

    plt.xlabel(f"Epoch ({length} steps each)")
    ylabel = "Average Estimated Value" if metric_type == "value" else "Average Reward"
    plt.ylabel(ylabel)
    plt.title(f"{strategy.upper()} Strategy - {ylabel} per Action")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{strategy}_{metric_type}_per_action.png"))
    plt.close()


# 主逻辑入口
all_results = {}

for strategy, filename in strategies.items():
    print(f"Processing {strategy}...")
    filepath = filename
    epoch_data = process_file(filepath)
    stats = compute_epoch_stats(epoch_data)
    all_results[strategy] = stats

    # 绘图：reward / value 按动作分类
    plot_action_metrics(stats, strategy, "reward")
    plot_action_metrics(stats, strategy, "value")

    # 汇总图：epoch 级别所有动作 reward 总和
    total_rewards = [sum(epoch[f"{action_names[a]}_reward"] for a in range(5)) for epoch in stats]
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(total_rewards) + 1), total_rewards, label=strategy.upper(), marker='o')
    plt.xlabel(f"Epoch ({length} steps each)")
    plt.ylabel("Total Avg Reward (All Actions)")
    plt.title(f"{strategy.upper()} Strategy - Total Avg Reward per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{strategy}_total_reward.png"))
    plt.close()

# 多策略对比图（按总 reward）
plt.figure(figsize=(12, 6))
for strategy, stats in all_results.items():
    total_rewards = [sum(epoch[f"{action_names[a]}_reward"] for a in range(5)) for epoch in stats]
    plt.plot(range(1, len(total_rewards) + 1), total_rewards, label=strategy.upper())
plt.xlabel(f"Epoch ({length} steps each)")
plt.ylabel("Total Avg Reward (All Actions)")
plt.title("Comparison of Total Avg Reward Across Strategies")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_strategies_total_reward_comparison.png"))
plt.show()

print("✅ 所有分析完成，图像已保存到 'analysis_results/' 文件夹中。")
