import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def generate_histograms_by_label(input_file, output_file, bin_width=0.10):

    data_by_label = defaultdict(list)

    with open(input_file, 'r') as file:
        for line in file:
            try:
                parts = line.strip().split(": ")[-1]
                parts2 = parts.split(maxsplit=1)
                print(parts2)
                # parts = line.strip().rsplit(" ", 1)  # Split into number and label
                # print(parts)
                # number = float(parts[-2])
                number = float(line.strip().split()[-2])
                print(number)
                # label = parts[-1]
                label= str(line.strip().split()[-1])
                data_by_label[label].append(number)
            except (ValueError, IndexError):
                print(f"Skipping invalid line: {line.strip()}")

    if not data_by_label:
        print("No valid data found in the file.")
        return

    plt.figure(figsize=(12, 8))
    for label, numbers in data_by_label.items():
        bins = np.arange(min(numbers), max(numbers) + bin_width, bin_width)
        plt.hist(numbers, bins=bins, alpha=0.5, label=f"Label: {label}", edgecolor='black')

    plt.title("Histograms by Label")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.grid(True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()

    print(f"Combined histogram saved to {output_file}")


input_file = "../logs/entropies.txt"
output_file = "../plots/individual_entropy_histograms.png"

generate_histograms_by_label(input_file, output_file)
