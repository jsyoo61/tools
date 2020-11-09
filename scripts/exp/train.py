import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Model

# 1. Load data
x_train = np.random.rand(200,10).astype(np.float32)

# 2. Hyperparameters
lr = 1e-4
batch_size = 150
epochs = 30

# 3. Create model
h_list = [4,4,4]
model = Model(input_size=8, output_size=3, h_list=h_list)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = DataLoader(x_train, batch_size=batch_size, shuffle=True) # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
criterion = nn.MSELoss()

# 4. Train
for epoch in range(1, epochs+1):

    for x, y in train_dataset:
        x = x.cuda()
        y = y.cuda()

        y_hat = model(x)
        loss = criterion(y_hat, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4-1. Tensorboard

        # 4-2. Print
        print(loss)

        # 4-3. Validation

    # Early stopping

    # Save intermediate model
    # Save only if validation loss has decreased

# 5. Save model
torch.save('model.pth', model.state_dict())
