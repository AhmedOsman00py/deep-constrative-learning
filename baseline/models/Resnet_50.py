import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Resnet_50_TL(nn.Module):
    def __init__(self, numbers_channels=1, numbers_classes=10):
        super(Resnet_50_TL, self).__init__()
        resnet_50 = models.resnet50(pretrained=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
