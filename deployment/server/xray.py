import torch
from torch import nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)  # Input shape (500, 500, 1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        
        # After convolutions and pooling, calculate the flattened size
        self.fc1 = nn.Linear(64 * 29 * 29, 128)  # This size may need adjustment based on input image size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Conv2D + ReLU
        x = self.pool(x)  # MaxPooling2D
        x = F.relu(self.conv2(x))  # Conv2D + ReLU
        x = self.pool(x)  # MaxPooling2D
        x = F.relu(self.conv3(x))  # Conv2D + ReLU
        x = self.pool(x)  # MaxPooling2D
        x = F.relu(self.conv4(x))  # Conv2D + ReLU
        x = self.pool(x)  # MaxPooling2D
        x = self.flatten(x)  # Flatten
        x = F.relu(self.fc1(x))  # Dense + ReLU
        x = F.relu(self.fc2(x))  # Dense + ReLU
        x = torch.sigmoid(self.fc3(x))  # Dense + Sigmoid
        return x 