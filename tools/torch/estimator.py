import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from sklearn.base import BaseEstimator
import sys
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NNEstimator(BaseEstimator):
    def __init__(self, model=None, criterion=None, val_dataset=None, epoch=10, lr=0.001, batch_size=32, patience=20, device=device, log_dir=None):
        # All arguments need default values. (sklearn)
        # All arguments must be fed into self.args = args
        self.model = model
        self.criterion = criterion

        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        # Optional arguments
        self.val_dataset = val_dataset
        self.log_dir = log_dir

    def fit(self, x, y):
        '''
        :param x: numpy.array
        :param y: nump.array
        '''
        self.model.to(self.device)
        op = optim.Adam(self.model.parameters(), lr=self.lr)

        x = torch.as_tensor(x, torch.float32)
        y = torch.as_tensor(y)

        loader = D.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epoch):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y, y_hat)

                op.zero_grad()
                loss.backward()
                op.step()

                if self.val_dataset is not None:
                    validation
                else:
                    loss.item()

        self.model.cpu()

        # if self.log_dir is None:
        #     log_dir = open(path.EXP.join(time.strftime('%Y-%m-%d_%H-%M-%S')+'.txt'))
        # default_stdout = sys.stdout
        # sys.stdout = log_dir


        # All parameters created here should be fed into self.args_ = args

    def predict(self, x):
        # return predicted value
        return y
