import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

def moving_average(data, window_size=3):
    """Apply a moving average to smooth the data."""
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    return smoothed

def plot_loss(log_file, plot_file="loss_plot.png",  smooth_window=5):
    epochs = []
    train_losses = []
    val_losses = []

    with open(log_file, "r") as log:
        next(log)
        for line in log:
            epoch, train_loss, val_loss = line.strip().split(",")
            epochs.append(int(epoch))
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

    #epochs = range(1, len(train_losses) + 1)

    smoothed_train_losses = moving_average(train_losses, window_size=smooth_window)
    smoothed_val_losses = moving_average(val_losses, window_size=smooth_window)

    smoothed_epochs = range(1 + (smooth_window - 1) // 2, len(train_losses) - (smooth_window - 1) // 2 + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(smoothed_epochs, smoothed_train_losses, label="Training Loss", marker="")
    plt.plot(smoothed_epochs, smoothed_val_losses, label="Validation Loss", marker="")
    plt.xlabel("Epochs")
    plt.ylabel(f"MSE Loss (Smooth-Moving Average w={smooth_window})")
    plt.title("[Obscurity Model] Training and Validation Loss")
    plt.legend()
    plt.grid()
    #plt.savefig(plot_file)  # Save the plot to a file
    print(f"Loss plot saved to {plot_file}")
    plt.show()


log_file = "../model/obscurity_training_w_clean_log_1.txt"

plot_file = "obscurity_loss_plot.png"
plot_loss(log_file, plot_file)