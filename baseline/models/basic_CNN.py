import torch
import torch.nn as nn
import torch.nn.functional as F


class basic_CNN(nn.Module):
    def __init__(self, numbers_channels=1, numbers_classes=10):
        super(basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=numbers_channels, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=32 * 14 * 14, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=numbers_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
