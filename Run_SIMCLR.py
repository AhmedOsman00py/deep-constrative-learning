import os
import torch
import torch.nn as nn 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

from baseline.utils import *
from baseline.training import training

from baseline.models.encoder import Encoder
from baseline.models.linear_classifier import LinearClassifier, TwoLayersClassifier
from baseline.models.simclr import SupCon

from pre_training_encoder import pretraining


class experiment():
    def __init__(self, numbers_exp, contrastive_loader, train_loader,val_loader):
        self.numbers_exp = numbers_exp
        self.contrastive_loader = contrastive_loader
        self.train_loader = train_loader
        self.val_loader = val_loader

    def pre_training(num_epochs, model, contrastive_loader, optimizer, criterion, use_scheduler = True):
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        contrastive_loss, contrastive_lr = [], []
        for epoch in range(1, num_epochs+1):
            loss, lr = pretraining(epochs, model, contrastive_loader, optimizer, criterion)
            if use_scheduler:
                scheduler.step()
            contrastive_loss.append(loss)
            contrastive_lr.append(lr)  
        plt.plot(range(1,len(contrastive_lr)+1),contrastive_lr, color='b', label = 'learning rate')
        plt.legend(), plt.ylabel('learning_rate'), plt.xlabel('epochs'), plt.title('Learning Rate'),
        plt.savefig("results/Learning rate.jpeg"), plt.show()
    
        plt.plot(range(1,len(contrastive_loss)+1),contrastive_loss, color='b', label = 'loss')
        plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'),
        plt.savefig("results/Loss function.jpeg"), plt.show()

        PATH = 'results/pre_trained_supcon.pth'
        torch.save(model.state_dict(), PATH)


