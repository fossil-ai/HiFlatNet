
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from obscurity import CNNRegression

import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F


import torch
import torchvision.models as models

# annoying, though GLU used weights... :(
class GLUWithWeights(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.glu(x)
        return x

class CNNFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(CNNFeatureExtractor, self).__init__()
        self.features = original_model.conv_layers
        self.model = original_model

    def forward(self, x):
        # Extract features from the final conv layer
        return self.features(x), self.model(x)

class ResNet18FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = torch.nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
        )
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def forward(self, x):
        feature_map = self.features(x)
        pooled_features = self.avgpool(feature_map)
        flattened = torch.flatten(pooled_features, 1)
        output = self.fc(flattened)
        return feature_map, output

class HiFlatNet(nn.Module):
    def __init__(self, obscurity_checkpoint_path, baseline_path, output_dim):
        super(HiFlatNet, self).__init__()

        output_dim = 2
        self.g = CNNRegression(output_dim=output_dim)
        self.g.load_state_dict(torch.load(obscurity_checkpoint_path))
        for param in self.g.parameters():
            param.requires_grad = False

        self.g_feature_extractor = CNNFeatureExtractor(self.g).to(device)

        self.h = models.resnet18(pretrained=True)
        num_ftrs = self.h.fc.in_features
        self.h.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
        self.h = self.h.to(device)

        self.h_feature_extractor = ResNet18FeatureExtractor(self.h).to(device)


        self.glu = GLUWithWeights(in_channels=32, out_channels=32)
        self.glu = self.glu.to(device)


        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Final classification head
        print("heh")
        print(self.h.fc.in_features)
        self.fc = nn.Linear(self.h.fc.in_features, 12)

        # Define normalization transform for the second model
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        resized_x1 = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        features_g, output_g = self.g_feature_extractor(resized_x1)

        normalized_x2 = torch.clone(x)
        for i in range(normalized_x2.size(0)):
            normalized_x2[i] = self.normalize(normalized_x2[i])
        features_h, output_h = self.h_feature_extractor(normalized_x2)


        glu_output = self.glu(features_g)

        upsample = nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False).to(device)
        upsampled_tensor = upsample(glu_output)

        # Step 2: Adjust channels from 32 to 512 using a 1x1 convolution
        conv = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1).to(device)
        aligned_tensor = conv(upsampled_tensor)

        combined_features = aligned_tensor + features_h
        gap_output = self.gap(combined_features)

        flattened_tensor = gap_output.view(x.shape[0], -1)

        out = self.fc(flattened_tensor)
        return out

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 10
output_dim = 10

data_dir = "../data/hiflatnet_training_data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

num_classes = len(train_dataset.classes)

class_to_idx = train_dataset.class_to_idx
print("Class to Index Mapping:", class_to_idx)
subset_classes = {0, 1, 2, 3, 5, 6, 7, 8, 9, 11}  # children classes, 4 and 10 are the parent classes

checkpoint1_path = "../checkpoints/obscurity_regression_model_w_clean.pth"
checkpoint2_path = "../checkpoints/baseline_resnet18_checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HiFlatNet(checkpoint1_path, checkpoint2_path, output_dim=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

num_epochs = 25
lambda_l2 = 1e-4

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)

        ce_loss = criterion(outputs, labels)

        l2_reg = 0.0
        for i in range(inputs.size(0)):  # Loop over the batch
            if predictions[i] == labels[i] and labels[i].item() in subset_classes: # check if correct AND in child class
                l2_reg += torch.sum(model.glu.conv.weight ** 2)

        total_loss = ce_loss + lambda_l2 * l2_reg

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

