import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageRegressionDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (str): Path to the text file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_path = parts[0]
                labels = np.array([float(x) for x in parts[1:]])  # Convert labels to float
                self.data.append((image_path, labels))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, labels = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)

class CNNRegression(nn.Module):
    def __init__(self, output_dim, reduced_channels=None):
        super(CNNRegression, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.reduced_channels = reduced_channels
        if reduced_channels:
            self.channel_reduction = nn.Conv2d(32, reduced_channels, kernel_size=1)
        else:
            self.channel_reduction = None
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear((reduced_channels or 32) * 16 * 16, 64),  # Assuming input images are 32x32
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        if self.channel_reduction:
            x = self.channel_reduction(x)  # Apply 1x1 convolution
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, patience=10, save_path="model.pth", log_file="loss_log.txt"):
    model.train()

    best_val_loss = float("inf")
    patience_counter = 0

    with open(log_file, "w") as log:
        log.write("epoch,train_loss,val_loss\n")  # Write header

        for epoch in range(num_epochs):
            running_loss = 0.0

            # Training loop
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)

                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_labels).item()

            val_loss /= len(val_loader)
            train_loss = running_loss / len(train_loader)
            print("-------------------------------------------------------------------------------")
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {running_loss / len(train_loader):.4f}, "
                f"Validation Loss: {val_loss:.4f}"
            )

            log.write(f"{epoch + 1},{train_loss},{val_loss}\n")

            print(f"Current Val: {val_loss} | Best Val: {best_val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"Validation loss improved to {best_val_loss}. Model saved to {save_path} at Epoch {epoch + 1}.")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience counter: {patience_counter}")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            model.train()


# 4. Main
if __name__ == "__main__":
    train_file_path = "../data/clear_aug_labels_train_w_clear.txt"
    val_file_path = "../data/clear_aug_labels_val_w_clear.txt"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageRegressionDataset(train_file_path, transform)
    val_dataset = ImageRegressionDataset(val_file_path, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = len(train_dataset[0][1])  # Infer the output dimension from the first sample's label
    print(output_dim)
    model = CNNRegression(output_dim=output_dim, reduced_channels=None).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00125)

    save_path = "obscurity_regression_model_w_clean_1.pth"
    log_file = "obscurity_training_w_clean_log_1.txt"
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=300, patience=40, save_path=save_path, log_file=log_file)
