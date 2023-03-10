import torch 
import torch.nn as nn 
from torch.nn import functional as F

class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 

        # Convolutional layers 
        self.conv1 = nn.Conv2d(3, 32, 3) # 3 input channels, 32 output channels, kernel size of 3x3 
        self.conv2 = nn.Conv2d(32, 64, 3) # 32 input channels, 64 output channels, kernel size of 3x3 

        # Max pooling layer 
        self.pool = nn.MaxPool2d(2) # pooling window of 2x2

        # Fully connected layers 
        self.fc1 = nn.Linear(64 * 8 * 8, 128) # 64 filters of 8x8 each flattened to 1D vector of length 128  
        self.fc2 = nn.Linear(128, 10) # 128 inputs to 10 outputs (classes)

    def forward(self, x): 

        x = self.pool(F.relu(self.conv1(x))) # convolutional layer 1 with ReLU activation followed by max pooling layer  
        x = self.pool(F.relu(self.conv2(x))) # convolutional layer 2 with ReLU activation followed by max pooling layer  

        x = x.view(-1, 64 * 8 * 8) # flatten the tensor for fully connected layers  

        x = F.relu(self.fc1(x)) # fully connected layer 1 with ReLU activation  
        x = F.softmax(self.fc2(x)) # fully connected layer 2 with softmax activation  

        return x
