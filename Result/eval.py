import json
from statistics import mean
from matplotlib import pyplot as plt


def extract_value_from_line(line, keyword):
    if keyword in line:
        try:
            start_index = line.index(keyword) + len(keyword) + 2
            end_index = line.index(',', start_index) if ',' in line[start_index:] else line.index('\n', start_index)
            value = line[start_index:end_index].strip()
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
                if value is not None and isinstance(value, (int, float)):
                    metrics[keyword].append(value)
    return metrics


def compare_files(file_paths):
    all_stats = []
    for file_path in file_paths:
        stats = process_file(file_path)
        all_stats.append((file_path, stats))

    plt.figure(figsize=(15, 10))

    for index, (file_path, stats) in enumerate(all_stats, start=1):
        plt.subplot(2, 1, index)
        plt.title(f"Data from {file_path}")

        for key, values in stats.items():
            if isinstance(values, list):
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

        plt.xticks(ticks=range(0, len(values), max(1, len(values) // 10)))
        plt.yticks(ticks=range(int(min(values)), int(max(values)), max(1, (int(max(values)) - int(min(values))) // 10)))
        plt.tight_layout()

        print(f"File: {file_path}")
        for key, values in stats.items():
            print(f"{key} Mean: {mean(values) if values else 'N/A'}")

    plt.savefig('chart.png', dpi=300)
    plt.show()


file_paths = ['sac_res.txt', 'dqn_res.txt']
compare_files(file_paths)
