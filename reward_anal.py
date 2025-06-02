import os
import re
import statistics
import chardet
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

# 文件名
filename = "output_greedy_v2.txt"

# 自动检测编码
with open(filename, "rb") as f:
    raw_data = f.read(1000)
    encoding = chardet.detect(raw_data)["encoding"]

print(f"文件编码: {encoding}")

# 正则匹配 [Step Reward] 行
reward_pattern = re.compile(r"'actions':\s*{([^}]+)}")

# 统计数据容器
action_rewards = {}
timestamps = []
cumulative_rewards = []
cumulative_reward = 0  # 初始化累计奖励为0

# 读取文件并解析
with open(filename, "r", encoding="utf-16") as file:
    for idx, line in enumerate(file):
        match = reward_pattern.search(line)
        if match:
            actions_str = match.group(1)
            actions = eval("{" + actions_str + "}")

            # 计算当前行所有动作奖励的总和
            total_reward_in_step = sum(actions.values())

            # 累计奖励
            cumulative_reward += total_reward_in_step

            timestamps.append(idx)
            cumulative_rewards.append(cumulative_reward)

            # 将数据存储到原本的动作奖励统计容器
            for action, reward in actions.items():
                if action not in action_rewards:
                    action_rewards[action] = []
                action_rewards[action].append(float(reward))

# 统计并输出
print(f"读取文件: {filename}\n")
print(f"共发现{len(action_rewards)}种不同的动作类型。\n")

action_stats = {}

for action, rewards in action_rewards.items():
    count = len(rewards)
    avg_reward = statistics.mean(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)
    median_reward = statistics.median(rewards)
    std_dev = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

    action_stats[action] = {
        "count": count,
        "avg": avg_reward,
        "min": min_reward,
        "max": max_reward,
        "median": median_reward,
        "std_dev": std_dev,
    }

    print(f"动作: {action}")
    print(f"  出现次数: {count}")
    print(f"  平均奖励: {avg_reward:.4f}")
    print(f"  最小奖励: {min_reward:.4f}")
    print(f"  最大奖励: {max_reward:.4f}")
    print(f"  中位数: {median_reward:.4f}")
    print(f"  奖励标准差: {std_dev:.4f}\n")

# 图像保存路径
output_dir = "report_images"
os.makedirs(output_dir, exist_ok=True)

# 1. 动作数量分布图
plt.figure(figsize=(12, 6))
actions = list(action_stats.keys())
counts = [action_stats[a]["count"] for a in actions]
plt.bar(actions, counts, color='skyblue')
plt.xlabel("Action")
plt.ylabel("Count")
plt.title("Action Occurrence Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "action_count.png"))
plt.close()

# 2. 每个动作的奖励分布图
plt.figure(figsize=(12, 6))
for action, rewards in action_rewards.items():
    plt.hist(rewards, bins=20, alpha=0.5, label=action)
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution by Action")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "reward_distribution.png"))
plt.close()

# 3. 每个动作的统计信息图
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.2
x = range(len(actions))

means = [action_stats[a]["avg"] for a in actions]
medians = [action_stats[a]["median"] for a in actions]
mins = [action_stats[a]["min"] for a in actions]
maxs = [action_stats[a]["max"] for a in actions]

ax.bar(x, means, width=bar_width, label='Mean', color='lightgreen')
ax.bar([p + bar_width for p in x], medians, width=bar_width, label='Median', color='lightskyblue')
ax.bar([p + 2 * bar_width for p in x], mins, width=bar_width, label='Min', color='lightcoral')
ax.bar([p + 3 * bar_width for p in x], maxs, width=bar_width, label='Max', color='gold')

ax.set_xlabel('Action')
ax.set_ylabel('Reward Value')
ax.set_title('Reward Statistics per Action')
ax.set_xticks([p + 1.5 * bar_width for p in x])
ax.set_xticklabels(actions, rotation=45)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "reward_stats.png"))
plt.close()

# 4. 每个动作奖励随时间变化的折线图
for action, rewards in action_rewards.items():
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(rewards)), rewards, label=action)
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title(f"Reward Change for Action: {action}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"reward_change_{action}.png"))
    plt.close()

# 5. 累计奖励折线图
plt.figure(figsize=(12, 6))
plt.plot(range(len(cumulative_rewards)), cumulative_rewards, label="Cumulative Reward", color="purple")
plt.xlabel("Time")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Time")
plt.legend()

# 动态调整纵坐标上限
max_cumulative_reward = max(cumulative_rewards)
plt.ylim(0, max_cumulative_reward * 1.1)  # 设置纵坐标范围为累计奖励的最大值的1.1倍
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cumulative_reward_all_actions.png"))
plt.close()

# 生成PDF报告
pdf_filename = "analysis_report.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

styles = getSampleStyleSheet()

caption_style = ParagraphStyle(
    name='Caption',
    parent=styles['Normal'],
    fontSize=10,
    textColor=colors.grey,
    spaceAfter=6,
    alignment=1
)

content = [Paragraph("Reward Analysis Report", styles['Title']), Spacer(1, 12)]

# 表格统计信息
table_data = [["Action", "Count", "Avg", "Min", "Max", "Median", "StdDev"]]
for action, stats in action_stats.items():
    table_data.append([
        action, stats["count"], f"{stats['avg']:.4f}", f"{stats['min']:.4f}",
        f"{stats['max']:.4f}", f"{stats['median']:.4f}", f"{stats['std_dev']:.4f}"
    ])
table = Table(table_data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER')
]))
content.append(table)
content.append(Spacer(1, 12))


# 图表部分
def add_image(caption, filename):
    content.append(Paragraph(caption, caption_style))
    content.append(Image(os.path.join(output_dir, filename), width=400, height=250))
    content.append(Spacer(1, 12))


# 累计奖励折线图
add_image("Figure: Cumulative Reward Over Time", "cumulative_reward_all_actions.png")

# 原有的图表
add_image("Figure 1: Action Occurrence Count", "action_count.png")
add_image("Figure 2: Reward Distribution by Action", "reward_distribution.png")
add_image("Figure 3: Reward Statistics per Action", "reward_stats.png")

# 每个动作的奖励随时间变化的折线图
for action in action_rewards:
    add_image(f"Figure: Reward Change for Action - {action}", f"reward_change_{action}.png")

doc.build(content)

print(f"分析报告已生成: {pdf_filename}")
