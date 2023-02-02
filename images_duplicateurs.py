## Bibliothèques 
from PIL import Image
from pathlib import Path
import IPython.display as display
import torch
from torchvision import transforms
import torchvision

## Classe des transfromations

class image_transformations :                    

    def __init__(self, image):
        self.image = image
    
    def data_augmentation1(self):
        Data_augmentation = torch.nn.Sequential(                      
                transforms.RandomHorizontalFlip(p=0.5),    
                transforms.RandomRotation(degrees=(-45, 45))
                     )
        return Data_augmentation(self.image)
        
    def data_augmentation2(self):
        Data_augmentation = torch.nn.Sequential(
                transforms.RandomPerspective(distortion_scale=0.6, p=1.0),     
                transforms.RandomVerticalFlip(p=0.5)
                     )
        return Data_augmentation(self.image)

## Exemple avec l'image de Messi

orig_img = Image.open('messi.jpg')
display.display(orig_img)

transform = image_transformations(orig_img)
Img1 = transform.data_augmentation1() #transformation 1
Img2 = transform.data_augmentation2() #transformation 2

# Affichage des resultats

display.display(Img1)
display.display(Img2)
