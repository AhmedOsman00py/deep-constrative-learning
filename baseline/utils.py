from torchvision import datasets
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def get_device():
    numbers_devices_available = torch.cuda.device_count()
    print(
        f" {torch.cuda.is_available} and can be used by {numbers_devices_available} devices"
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"The device used is {torch.cuda.get_device_name(0)}")
    return device


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_data, test_data


def get_loader(train_data, test_data, batch_size=32):
    random_state = 2023
    unlabeled_indices, labeled_indices = train_test_split(
        list(range(len(train_data.targets))),
        test_size=100,
        stratify=train_data.targets,
        random_state=random_state,
    )
    unlabeledset = torch.utils.data.Subset(train_data, unlabeled_indices)
    labeledset = torch.utils.data.Subset(train_data, labeled_indices)
    train_unlabeled_loader = torch.utils.data.DataLoader(
        unlabeledset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    train_labeled_loader = torch.utils.data.DataLoader(
        labeledset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_unlabeled_loader, train_labeled_loader, test_loader


def get_accuracy(y_true, y_pred):
    return int(np.sum(np.equal(y_true, y_pred))) / y_true.shape[0]

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False