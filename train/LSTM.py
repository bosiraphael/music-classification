import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

NUM_CLASSES = 10
NUM_LAYERS = 1
INPUT_SIZE = 20
HIDDEN_SIZE = 50

class LSTMNET(nn.Module):
    def __init__(self, dropout):
        super(LSTMNET, self).__init__()
        self.num_classes = NUM_CLASSES #number of classes
        self.num_layers = NUM_LAYERS #number of layers
        self.input_size = INPUT_SIZE #input size
        self.hidden_size = HIDDEN_SIZE #hidden state
        self.dropout = nn.Dropout(p = dropout)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True) #lstm
        self.fc1 = nn.Linear(self.hidden_size * self.num_layers, self.num_classes) #fully connected layer
        self.fc2 = nn.Linear(self.hidden_size * self.num_layers, self.num_classes) #fully connected layer
        
        self.relu = nn.ReLU()
    
    def forward(self,x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device).detach_() #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device).detach_() #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size * self.num_layers) #reshaping the data for Dense layer next
        out = self.dropout(hn)
        out = self.relu(hn)
        out = self.fc1(out)
        out = self.relu(hn)
        out = self.fc2(out)
        return out