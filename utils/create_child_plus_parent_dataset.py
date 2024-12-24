import json
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


def allocate(ontology, entropy_thresholds, checkpoint_path, data_dir):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    # def evaluate(dataset, dataset_name, output_dir):
    #     "
    #     all_preds = []
    #     all_labels = []
    #     all_softmax = []
    #     all_images = []
    #
    #     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    #     output_plot_dir = os.path.join(output_dir, dataset_name)
    #     os.makedirs(output_plot_dir, exist_ok=True)
    #
    #     with torch.no_grad():
    #         for idx, (inputs, labels) in enumerate(loader):
    #             inputs, labels = inputs.to(device), labels.to(device)
    #
    #             outputs = model(inputs)
    #             softmax_outputs = torch.softmax(outputs, dim=1).cpu().numpy()
    #
    #             all_preds.append(outputs.argmax(dim=1).cpu().item())
    #             all_labels.append(labels.cpu().item())
    #             all_softmax.append(softmax_outputs[0])
    #             all_images.append(inputs.cpu().numpy()[0])
    #
    #             class_label = dataset.classes[labels.cpu().item()]
    #             class_dir = os.path.join(output_plot_dir, class_label)
    #             os.makedirs(class_dir, exist_ok=True)
    #
    #             entropy = -np.sum(softmax_outputs[0] * np.log(softmax_outputs[0] + 1e-9))
    #             print(f"{dataset_name} sample {idx + 1} with entropy: {entropy:.4f} {class_label}")
    def evaluate(dataset, dataset_name, output_dir, entropy_threshold):
        """Evaluate and save results."""
        all_preds = []
        all_labels = []
        all_softmax = []
        all_images = []

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        output_plot_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(output_plot_dir, exist_ok=True)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        def unnormalize_image(tensor_image):
            """Unnormalize an image tensor and convert to RGB format."""
            img = tensor_image.cpu().numpy().transpose((1, 2, 0))
            img = (img * std + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            return img

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                softmax_outputs = torch.softmax(outputs, dim=1).cpu().numpy()

                pred_label = outputs.argmax(dim=1).cpu().item()
                true_label = labels.cpu().item()

                all_preds.append(pred_label)
                all_labels.append(true_label)
                all_softmax.append(softmax_outputs[0])
                all_images.append(inputs.cpu().numpy()[0])

                # Compute entropy
                entropy = -np.sum(softmax_outputs[0] * np.log(softmax_outputs[0] + 1e-9))
                class_label = dataset.classes[true_label]
                pred_class_label = dataset.classes[pred_label]

                is_correct = pred_label == true_label

                gamma = entropy_threshold[class_label]
                # Determine the save directory based on entropy
                if entropy > gamma or not is_correct:
                    class_label = ontology[class_label]

                class_dir = os.path.join(output_plot_dir, class_label)
                os.makedirs(class_dir, exist_ok=True)

                #img = inputs.cpu().numpy()[0].transpose((1, 2, 0))
                img = unnormalize_image(inputs[0])
                # img = (img * 255).astype(np.uint8)
                img_path = os.path.join(class_dir, f"sample_{idx + 1}.png")
                plt.imsave(img_path, img)

                print(f"{dataset_name} sample {idx + 1} with entropy: {entropy:.4f} saved to {class_label}")

    evaluate(train_dataset, "train", output_dir="test_parent", entropy_threshold=entropy_thresholds)
    evaluate(test_dataset, "train", output_dir="test_parent", entropy_threshold=entropy_thresholds)


input_directory = "../data/baseline_training_data"

with open("../data/entropy_thresholds.json", "r") as entropy_map_file:
    thresholds = json.load(entropy_map_file)

with open("../data/ontology.json", "r") as ontology_file:
    ontology = json.load(ontology_file)

allocate(
    ontology=ontology,
    entropy_thresholds = thresholds,
    checkpoint_path="../checkpoints/baseline_resnet18_checkpoint.pth",
    data_dir=input_directory,
)
