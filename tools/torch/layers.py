import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class attention(nn.Module):
    def __init__(self, dim=1):
        super(att_pool, self).__init__()
        self.attention = nn.softmax()


    def forward(self, x):
        return x*self.attention(x)
