from torchvision import datasets
from torchvision import transforms
import torch


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_contrastive_loader(batch_size = 64):
    contrastive_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    contrastive_set = datasets.MNIST(
        "./data",
        download=True,
        train=True,
        transform=TwoCropTransform(contrastive_transform),
    )
    contrastive_loader = torch.utils.data.DataLoader(
        contrastive_set, batch_size=batch_size, shuffle=True
    )
    return contrastive_loader
