import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Model

model = Model()


state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)
model.cuda()
