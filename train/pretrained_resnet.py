import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = Net(num_ftrs)
    
    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
      for param in self.network.parameters():
        param.require_grad = False
      for param in self.network.fc.parameters():
        param.require_grad = True
    
    def unfreeze(self):
      for param in self.network.parameters():
        param.require_grad = True

class Net(nn.Module):
    def __init__(self, num_ftrs):
        super().__init__()
        self.linear1 = nn.Linear(num_ftrs, 50)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(50, 10)
    
    def forward(self, xb):
        out = self.linear1(xb)
        out = self.relu(out)
        out = self.dropout(out)
        return self.linear2(out)