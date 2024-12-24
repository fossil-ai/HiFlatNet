import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

entropy_threshold = {}

def generate_histograms_by_label(input_file, output_file, bin_width=0.05):

    data_by_label = defaultdict(list)

    try:
        with open(input_file, 'r') as file:
            for line in file:
                try:
                    # Parse the line for number and label
                    parts = line.strip().split(": ")
                    parts = parts[-1].split(maxsplit=1)
                    number = float(parts[-2])
                    label = parts[-1]
                    data_by_label[label].append(number)
                except (ValueError, IndexError):
                    print(f"Skipping invalid line: {line.strip()}")
    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
        return

    if not data_by_label:
        print("No valid data found in the file.")
        return

    labels = list(data_by_label.keys())
    num_labels = len(labels)

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, label in enumerate(labels):
        if i >= 10:
            break
        ax = axes[i]
        numbers = data_by_label[label]
        bins = np.arange(min(numbers), max(numbers) + bin_width, bin_width)
        ax.hist(numbers, bins=bins, alpha=0.7, color='blue', edgecolor='black')

        # Calculate the 75th percentile and add a red vertical line
        percentile_75 = np.percentile(numbers, 75)
        ax.axvline(percentile_75, color='red', linestyle='--', linewidth=1.5, label='75th Percentile')

        ax.set_title(f"{label}", fontsize=20)
        ax.set_xlabel("Entropy", fontsize=15)
        ax.set_ylabel("Frequency", fontsize=15)

        entropy_threshold[label] = percentile_75


    for j in range(len(labels), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Combined histogram saved to {output_file}")

input_file = "../logs/entropies.txt"
output_file = "../plots/combined_histograms.png"

generate_histograms_by_label(input_file, output_file)

print(entropy_threshold)