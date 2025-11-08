import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# ========= 配置 =========
LOG_FILE = "mcts_result.txt"
ACTIONS = ["Illegal", "Wait", "Withhold", "Adopt", "Release", "Equivocate", "Exit"]

# ========= 正则匹配 =========
pattern_action = re.compile(
    r"\[(MCTS|PPO|GREEDY|Random) Decision\].*Chosen Action:\s*(\d+).*?(?:Estimated Value|Log Prob|Q-value)[:=]\s*([-+]?\d*\.\d+|\d+)"
)
pattern_reward = re.compile(
    r"\[Env Feedback (MCTS|PPO|GREEDY|RANDOM)\].*Reward:\s*([-+]?\d*\.\d+|\d+)"
)

# ========= 数据结构 =========
data = {
    "MCTS": {"actions": [], "values": [], "rewards": []},
    "PPO": {"actions": [], "values": [], "rewards": []},
    "GREEDY": {"actions": [], "values": [], "rewards": []},
    "RANDOM": {"actions": [], "values": [], "rewards": []},
}

# ========= 读取日志文件 =========
with open(LOG_FILE, "r", encoding="utf-16") as f:
    for line in f:
        line = line.strip()

        # 匹配选择动作
        match_a = pattern_action.search(line)
        if match_a:
            strategy, action, value = match_a.groups()
            strategy = strategy.upper()
            data[strategy]["actions"].append(int(action))
            data[strategy]["values"].append(float(value))
            continue

        # 匹配奖励
        match_r = pattern_reward.search(line)
        if match_r:
            strategy, reward = match_r.groups()
            strategy = strategy.upper()
            data[strategy]["rewards"].append(float(reward))
            continue

# ========= 分析函数 =========
def analyze_strategy(strategy: str, records: dict):
    actions = records["actions"]
    rewards = records["rewards"]
    values = records["values"]

    if not actions:
        print(f"[WARN] No data for {strategy}")
        return None

    total_steps = len(actions)
    total_reward = np.sum(rewards) if rewards else 0.0
    avg_reward = np.mean(rewards) if rewards else 0.0
    avg_value = np.mean(values) if values else 0.0

    # 每个动作的统计
    action_counts = Counter(actions)
    avg_reward_per_action = defaultdict(float)
    if rewards:
        # 简单平均每步reward到各动作（近似统计）
        for i, a in enumerate(actions):
            avg_reward_per_action[a] += rewards[i] if i < len(rewards) else 0
        for a in avg_reward_per_action:
            avg_reward_per_action[a] /= action_counts[a]

    print(f"\n===== {strategy} 统计结果 =====")
    print(f"总步数: {total_steps}")
    print(f"总奖励: {total_reward:.4f}")
    print(f"平均每步奖励: {avg_reward:.4f}")
    print(f"平均估值: {avg_value:.4f}")
    print(f"动作分布:")
    for a in range(7):
        cnt = action_counts[a]
        ar = avg_reward_per_action.get(a, 0.0)
        print(f"  - {ACTIONS[a]:<10s} 次数={cnt:<5d} 平均奖励={ar:.4f}")

    return {
        "total_steps": total_steps,
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "action_counts": action_counts,
        "avg_reward_per_action": avg_reward_per_action,
    }

# ========= 结果统计 =========
results = {}
for strat, rec in data.items():
    res = analyze_strategy(strat, rec)
    if res:
        results[strat] = res

# ========= 可视化 =========
plt.figure(figsize=(14, 8))
for strat, res in results.items():
    plt.plot(np.arange(res["total_steps"]), data[strat]["rewards"][:res["total_steps"]],
             label=f"{strat} (Avg {res['avg_reward']:.3f})", alpha=0.7)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Progression per Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_curve.png")
plt.show()

# -------- 动作分布图 --------
plt.figure(figsize=(10, 6))
bar_width = 0.2
x = np.arange(len(ACTIONS))
for i, (strat, res) in enumerate(results.items()):
    counts = [res["action_counts"].get(a, 0) for a in range(7)]
    plt.bar(x + i * bar_width, counts, width=bar_width, label=strat)
plt.xticks(x + bar_width * 1.5, ACTIONS, rotation=30)
plt.title("Action Frequency by Strategy")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("action_distribution.png")
plt.show()

# -------- 动作平均奖励图 --------
plt.figure(figsize=(10, 6))
for strat, res in results.items():
    avg_rewards = [res["avg_reward_per_action"].get(a, 0) for a in range(7)]
    plt.plot(ACTIONS, avg_rewards, marker="o", label=strat)
plt.title("Average Reward per Action")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("action_avg_reward.png")
plt.show()
