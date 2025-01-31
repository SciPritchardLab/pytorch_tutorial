import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Convolution + ReLU + MaxPool (1st layer)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # Convolution + ReLU + MaxPool (2nd layer)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the output for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x