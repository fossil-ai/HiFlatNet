import matplotlib.pyplot as plt

def extract_losses(file_path):
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            loss = float(line.strip().split(" ")[-1])
            losses.append(loss)
    return losses

file_1 = '../logs/with_obs_module.txt'
file_2 = '../logs/without_obs_module.txt'

losses_1 = extract_losses(file_1)
losses_2 = extract_losses(file_2)

plt.figure(figsize=(10, 6))

plt.plot(losses_1, label='w/ Obscurity Module, L2=1e-4', color='b', linestyle='-', marker='o')

plt.plot(losses_2, label='w/ out Obscurity Module, L2=1e-4', color='r', linestyle='-', marker='x')

plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.title('Training Loss Comparison With and Without the Obscurity Guidance', fontsize=20)

plt.legend()
plt.show()
