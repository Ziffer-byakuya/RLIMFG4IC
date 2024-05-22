import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F


class RND(nn.Module):
    def __init__(self, num_inputs, scale):
        super(RND, self).__init__()
        self.hidden = 512
        self.scale = scale
        self.linear1 = nn.Linear(num_inputs, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.linear3 = nn.Linear(self.hidden, self.hidden)
        self.linear4 = nn.Linear(self.hidden, self.hidden)
        self.linear5 = nn.Linear(self.hidden, 32)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.tanh(x)*self.scale
        return x
