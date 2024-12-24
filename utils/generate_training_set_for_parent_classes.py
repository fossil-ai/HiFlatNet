import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def evaluate_checkpoint(checkpoint_path, data_dir):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    def evaluate(dataset, dataset_name, output_dir):
        all_preds = []
        all_labels = []
        all_softmax = []
        all_images = []

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        output_plot_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(output_plot_dir, exist_ok=True)

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                softmax_outputs = torch.softmax(outputs, dim=1).cpu().numpy()

                all_preds.append(outputs.argmax(dim=1).cpu().item())
                all_labels.append(labels.cpu().item())
                all_softmax.append(softmax_outputs[0])
                all_images.append(inputs.cpu().numpy()[0])

                # Save plot
                class_label = dataset.classes[labels.cpu().item()]
                # class_dir = os.path.join(output_plot_dir, class_label)
                # os.makedirs(class_dir, exist_ok=True)
                #
                # plt.figure(figsize=(12, 4))
                #
                # # Plot the image
                # plt.subplot(1, 2, 1)
                # image = np.transpose(inputs.cpu().numpy()[0], (1, 2, 0))  # Convert to HWC format
                # image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
                # image = np.clip(image, 0, 1)  # Clip values to [0, 1]
                # plt.imshow(image)
                # plt.axis('off')
                # plt.title(f"Image")
                #
                # # Plot the softmax distribution
                # plt.subplot(1, 2, 2)
                # plt.bar(range(len(softmax_outputs[0])), softmax_outputs[0])
                # plt.title("Softmax Distribution")
                # plt.xlabel("Class")
                # plt.ylabel("Probability")
                #
                # plt.tight_layout()
                #
                # # Save the plot
                # plot_path = os.path.join(class_dir, f"{dataset_name}_sample_{idx + 1}.png")
                # plt.savefig(plot_path)
                # plt.close()

                entropy = -np.sum(softmax_outputs[0] * np.log(softmax_outputs[0] + 1e-9))
                print(f"{dataset_name} sample {idx + 1} with entropy: {entropy:.4f} {class_label}")

    # print("Evaluating on Training Set")
    # evaluate(train_dataset, "Training")
    #
    # print("Evaluating on Validation Set")
    # evaluate(test_dataset, "Validation")

    evaluate(train_dataset, "Training", output_dir="evaluation_results")
    evaluate(test_dataset, "Validation", output_dir="evaluation_results")


output_directory = "../data/baseline_training_data"

evaluate_checkpoint(
    checkpoint_path="../checkpoints/baseline_resnet18_checkpoint.pth",
    data_dir=output_directory,
)
