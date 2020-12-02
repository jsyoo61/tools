import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DNN(nn.Module):
    '''
    basic DNN module

    Parameters
    ----------
    n_input: number of input (int)
    n_hidden_list: list of hidden neurons (list of int)
    activation_list: torch.nn activation function instances (nn activation instance or list)

    Example
    -------
    >>> model = DNN(n_input=10, n_hidden_list=[8,6,5], activation_list=[nn.Sigmoid(), nn.ReLU(), nn.Tanh()])
    # n_hidden_list, activation_list corresponds to [h1, h2, output]
    >>> print(model)
    DNN(
      (fc): Sequential(
        (0): Linear(in_features=10, out_features=8, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=8, out_features=6, bias=True)
        (3): ReLU()
        (4): Linear(in_features=6, out_features=5, bias=True)
        (5): Tanh()
      )
    )
    '''
    def __init__(self, n_input, n_hidden_list, activation_list):
        super(DNN, self).__init__()
        if type(activation_list) is not list:
            activation_list = [activation_list]*len(n_hidden_list)
        assert len(activation_list)==len(n_hidden_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'

        # 1st layer
        layers = [nn.Linear(n_input, n_hidden_list[0]), activation_list[0]]
        # Hidden layers ~ Output layer
        for i in range(len(n_hidden_list) - 1):
            layers.extend([nn.Linear(n_hidden_list[i], n_hidden_list[i+1]), activation_list[i+1]])

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        '''x.shape()==(batch_size, feature_dim)'''
        return self.fc(x)

def residual_computation(sequential, x, connection='sequential'):
    if connection=='sequential':
        pass
    elif connection=='dense':
        pass

    return

if __name__ == '__main__':
    n_input = 5
    n_hidden_list = [10,20,30,20,10]
    activation_lisy = [nn.Sigmoid(), nn.ReLU(), nn.LeakyReLU(0.03), nn.Tanh(), nn.Identity()]
    model = DNN(n_input, n_hidden_list, activation_lisy)
    print(model)
