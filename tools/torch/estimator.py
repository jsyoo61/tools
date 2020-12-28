import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from model import DNN
import sys
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class estimator(BaseEstimator):
    def __init__(self, model=DNN(10, [5,3,1], nn.ReLU()), val_dataset='', n_epoch=10, lr=0.001, batch_size=32, device=device, log_dir=None):
        # All arguments need default values.
        # All arguments must be fed into self.args = args
        self.model = model
        self.val_dataset = val_dataset
        self.n_epoch = n_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.log_dir = log_dir

    def fit(self, x, y):
        if self.log_dir is None:
            log_dir = open(path.EXP.join(time.strftime('%Y-%m-%d_%H-%M-%S')+'.txt'))
        default_stdout = sys.stdout
        sys.stdout = log_dir


        # All parameters created here should be fed into self.args_ = args


    def predict(self, x):
        # return predicted value
        return y
