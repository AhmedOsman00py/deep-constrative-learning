# Bibliothèques
import torch
from torchvision import transforms


class image_transformations:
    def __init__(self, image):
        self.image = image

    def data_augmentation1(self):
        Data_augmentation = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.ElasticTransform(alpha=250.0),
        )
        return Data_augmentation(self.image)

    def data_augmentation2(self):
        Data_augmentation = torch.nn.Sequential(
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        )
        return Data_augmentation(self.image)