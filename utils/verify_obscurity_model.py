# import torch
# from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from model.obscurity import CNNRegression

import matplotlib.pyplot as plt
import torch


def plot_feature_maps(features, save_path=None):

    assert features.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
    assert features.shape[0] == 1, "Batch size must be 1 for visualization"

    features = features.squeeze(0)

    num_channels = features.shape[0]
    grid_size = int(num_channels ** 0.5) + (1 if num_channels ** 0.5 % 1 else 0)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            feature_map = features[i].detach().cpu().numpy()
            ax.imshow(feature_map, cmap="viridis")
            ax.set_title(f"Channel {i + 1}")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Feature maps saved to {save_path}")
    plt.show()


class CNNFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(CNNFeatureExtractor, self).__init__()
        self.features = original_model

    def forward(self, x):
        return self.features(x)

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def extract_features(image_path, model_path, transform):
    output_dim = 2
    original_model = CNNRegression(output_dim=output_dim)
    original_model.load_state_dict(torch.load(model_path))
    original_model.eval()

    feature_extractor = CNNFeatureExtractor(original_model)

    input_image = preprocess_image(image_path, transform)

    with torch.no_grad():
        features = feature_extractor(input_image)

    return features
nonaug = "/home/mohamf1-ll1/Desktop/School/robustml/HiFlatClassifier/data/original/SUV/598d4732-ed72-4cfd-899d-5a560c5f9930_frame-68.000000_1.jpg"
#Extracted features: tensor([[1.7755, 1.0854]])

aug = "/home/mohamf1-ll1/Desktop/School/robustml/HiFlatClassifier/data/clear_augmented_w_clear/2e24b20f-2c55-48cb-9ef4-644eee531773_frame-104.000000_0_aug_31.jpg"


# real nonaug withblur /home/mohamf1-ll1/Desktop/School/robustml/HiFlatClassifier/data/original/SUV/75dfd66d-a22d-4981-aea4-813e6c370ce6_frame-36.000000_0.jpg
# real nonaug blur and hue /home/mohamf1-ll1/Desktop/School/robustml/HiFlatClassifier/data/original/SUV/7259a5f4-92c6-4493-9bb6-5d17a7ec8945_frame-0.000000_0.jpg
# Main for extracting features
if __name__ == "__main__":
    model_path = "../checkpoints/obscurity_regression_model_w_clean.pth"
    image_path = nonaug

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    features = extract_features(image_path, model_path, transform)

    print(f"Extracted features shape: {features.shape}")
    print(f"Extracted features: {features}")

