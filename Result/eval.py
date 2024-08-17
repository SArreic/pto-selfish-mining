import json
from statistics import mean
from matplotlib import pyplot as plt


def extract_value_from_line(line, keyword):
    if keyword in line:
        try:
            start_index = line.index(keyword) + len(keyword) + 2
            end_index = line.index(',', start_index) if ',' in line[start_index:] else line.index('\n', start_index)
            value = line[start_index:end_index].strip()

            if '±' in value:
                mean_value, error_value = value.split('±')
                return float(mean_value), float(error_value)

            if value.startswith('{'):
                return json.loads(value)

            if value.replace('.', '', 1).isdigit():
                return float(value) if '.' in value else int(value)

            return value
        except ValueError as e:
            print(f"Error extracting value for {keyword}: {e}")
            print(f"Faulty line: {line}")
    return None


def process_file(file_path):
    metrics = {
        'Base Value Approximation': [],
        'Test Policy Revenue': [],
        'Simulated Policy Revenue': [],
        'Agent Base Value Approximation': []
    }

    with open(file_path, 'r', encoding='utf-16') as file:
        for line in file:
            for keyword in metrics:
                value = extract_value_from_line(line, keyword)
                if value is not None:
                    if isinstance(value, tuple):
                        metrics[keyword].append(value[0])  # 只保存均值
                    elif isinstance(value, (int, float)):
                        metrics[keyword].append(value)
    return metrics


def plot_comparison(all_stats, keyword):
    plt.figure(figsize=(15, 10))
    plt.title(f"Comparison of {keyword}")

    for file_path, stats in all_stats:
        values = stats[keyword]
        if isinstance(values, list) and values:
            if len(values) > 10:
                subset_values = values[::10]
                plt.plot(subset_values, label=f'{keyword} from {file_path}', marker='o', linestyle='-', markersize=4)
            else:
                plt.plot(values, label=f'{keyword} from {file_path}', marker='o', linestyle='-', markersize=4)

    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'comparison_{keyword}.png', dpi=300)
    plt.show()


def compare_files(file_paths):
    all_stats = []
    for file_path in file_paths:
        stats = process_file(file_path)
        all_stats.append((file_path, stats))

    # Iterate over each file and its corresponding stats
    for file_path, stats in all_stats:
        for key, values in stats.items():
            if key == 'Simulated Policy Revenue':
                print(f"File: {file_path}")
                print(f"{key} Mean: {mean(values) if values else 'N/A'}")
                continue  # Skip plotting for Simulated Policy Revenue

            plt.figure(figsize=(15, 10))
            plt.title(f"{key} Data from {file_path}")

            if isinstance(values, list) and values:
                if len(values) > 10:
                    subset_values = values[::10]
                    plt.plot(subset_values, label=key, marker='o', linestyle='-', markersize=4)
                else:
                    plt.plot(values, label=key, marker='o', linestyle='-', markersize=4)
            else:
                plt.plot([values], label=key, marker='o', linestyle='-', markersize=4)

            plt.xlabel('Data Points')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)

            # Set the xticks and yticks as specified
            max_x = len(values)
            if key == 'Test Policy Revenue':
                plt.xticks(ticks=range(0, 501, 50))
                plt.yticks(ticks=[i * 0.2 for i in range(0, int(1 / 0.2) + 1)])
                plt.xlim(0, 500)  # Set the x-axis limit for Test Policy Revenue
                plt.ylim(0, 1)  # Set the y-axis limit for Test Policy Revenue
            else:
                plt.xticks(ticks=range(0, max_x, 500))
                plt.yticks(ticks=[i * 0.1 for i in range(0, int(5000 / 0.1) + 1)])
                plt.xlim(0, 4000)  # Set the x-axis limit
                plt.ylim(0, 0.5)

            plt.tight_layout()

            plt.savefig(f'chart_{file_path}_{key}.png', dpi=300)
            plt.show()

            print(f"File: {file_path}")
            print(f"{key} Mean: {mean(values) if values else 'N/A'}")

    # Plot comparisons for Agent Base Value Approximation and Base Value Approximation
    plot_comparison(all_stats, 'Agent Base Value Approximation')
    plot_comparison(all_stats, 'Base Value Approximation')


file_paths = []
compare_files(file_paths)
