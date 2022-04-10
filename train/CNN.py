import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CONV_1 = 8
NUM_CONV_2 = 16
NUM_CONV_3 = 32
NUM_CONV_4 = 64
NUM_CONV_5 = 128

NUM_FC = 500
NUM_CLASSES = 10

class CNNNet(nn.Module):
    
    def __init__(self, dropout):
        super(CNNNet,self).__init__()
        self.conv_1 = nn.Conv2d(1,NUM_CONV_1,5,1) # kernel_size = 5
        self.conv_2 = nn.Conv2d(NUM_CONV_1,NUM_CONV_2,5,1) # kernel_size = 5

        self.batch_norm_1 = nn.BatchNorm2d(NUM_CONV_1)
        self.batch_norm_2 = nn.BatchNorm2d(NUM_CONV_2)

        self.drop = nn.Dropout2d(p=dropout)

        self.fc_1 = nn.Linear(51*2*NUM_CONV_2, NUM_FC)
        self.fc_2 = nn.Linear(NUM_FC,NUM_CLASSES)
    
    def forward(self,x):
        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.batch_norm_2(self.conv_2(x)))
        x = F.max_pool2d(x,2,2)
        #print(x.shape)
        x = x.view(-1, 51 * 2 * NUM_CONV_2)
        #print(x.shape)
        x = self.drop(x)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x