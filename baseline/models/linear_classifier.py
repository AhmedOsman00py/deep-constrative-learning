import torch


class LinearClassifier(torch.nn.Module):
    """Linear classifier"""

    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 128, 10),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TwoLayersClassifier(torch.nn.Module):
    """Linear classifier"""

    def __init__(self):
        super(TwoLayersClassifier, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 128, 512), torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
