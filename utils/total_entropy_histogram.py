import matplotlib.pyplot as plt
import numpy as np
import os

def generate_histogram(input_file, output_file, bin_width=0.04):

    numbers = []
    with open(input_file, 'r') as file:
        for line in file:
            try:
                # Extract the last number in the line
                number = float(line.strip().split()[-2])
                print(number)
                numbers.append(number)
            except (ValueError, IndexError):
                print(f"Skipping invalid line: {line.strip()}")

    if not numbers:
        print("No valid numbers found in the file.")
        return

    bins = np.arange(min(numbers), max(numbers) + bin_width, bin_width)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(numbers, bins=bins, edgecolor='black', alpha=0.7)
    plt.title("Histogram of Numbers")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the histogram
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()

    print(f"Histogram saved to {output_file}")


input_file = "../logs/entropies.txt"
output_file = "../plots/entropy_histogram.png"
generate_histogram(input_file, output_file)
