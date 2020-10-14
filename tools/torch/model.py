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
    n_output: number of output (int)
    n_hidden_list: list of hidden neurons (list of int)
    activation: torch.nn activation function instances (nn activation instance or list)
    '''
    def __init__(self, n_input, n_output, n_hidden_list, activation):
        super(DNN, self).__init__()
        if type(activation) is not list:
            activation = [activation]*(len(n_hidden_list)+1)
        assert len(activation)==len(n_hidden_list)+1, 'length of layers and activations must match. If you want no activation, use nn.Identity'

        layers = [nn.Linear(n_input, n_hidden_list[0]), activation[0]]
        for i in range(len(n_hidden_list) - 1):
            layers.extend([nn.Linear(n_hidden_list[i], n_hidden_list[i+1]), activation[i+1]])
        layers.extend([nn.Linear(n_hidden_list[-1], n_output), activation[-1]])

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

if __name__ == '__main__':
    n_input = 5
    n_output = 10
    n_hidden_list = [10,20,30,20]
    activation = [nn.Sigmoid(), nn.ReLU(), nn.LeakyReLU(0.03), nn.Tanh(), nn.Identity()]
    model = DNN(n_input, n_output, n_hidden_list, activation)
    print(model)
