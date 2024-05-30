import json
from statistics import mean

from matplotlib import pyplot as plt


def extract_value_from_line(line, keyword):
    if keyword in line:
        try:
            # Assuming the value always follows the keyword and a colon, and ends with a newline or comma.
            start_index = line.index(keyword) + len(keyword) + 2
            end_index = line.index(',', start_index) if ',' in line[start_index:] else line.index('\n', start_index)
            value = line[start_index:end_index].strip()
            # If the value is a JSON object, parse it
            if value.startswith('{'):
                return json.loads(value)
            # If the value is a number, convert it
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
        # Add other metrics as needed
    }

    with open(file_path, 'r', encoding='utf-16') as file:
        for line in file:
            for keyword in metrics:
                value = extract_value_from_line(line, keyword)
                if value is not None:
                    metrics[keyword].append(value)
    return metrics


def compare_files(file_paths):
    all_stats = []
    for file_path in file_paths:
        stats = process_file(file_path)
        all_stats.append((file_path, stats))

    plt.figure(figsize=(20, 10 * len(file_paths)))

    for index, (file_path, stats) in enumerate(all_stats, start=1):
        plt.subplot(len(file_paths), 1, index)
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
        plt.tight_layout()

    plt.savefig('chart.png', dpi=300)
    plt.show()


# List of files to compare
file_paths = ['sac_res.txt', 'dqn_res.txt', 'dqn_res_1_step.txt']
compare_files(file_paths)
