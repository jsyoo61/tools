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
    )-
    '''
    def __init__(self, n_input, n_hidden_list, activation_list):
        super(DNN, self).__init__()
        if type(activation_list) is not list:
            activation_list = [activation_list]*len(n_hidden_list)
        assert len(activation_list)==len(n_hidden_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'

        # 1st layer
        layers = [nn.Flatten(1), nn.Linear(n_input, n_hidden_list[0]), activation_list[0]]
        # Hidden layers ~ Output layer
        for i in range(len(n_hidden_list) - 1):
            layers.extend([nn.Linear(n_hidden_list[i], n_hidden_list[i+1]), activation_list[i+1]])

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        '''x.shape()==(batch_size, feature_dim)'''
        return self.fc(x)

class DNN_Resnet(nn.Module):
    '''
    Resnet module

    Parameters
    ----------
    n_input : int
        number of input
    n_hidden_list : list of int
        list of hidden neurons
    activation : nn activation instance or list
        torch.nn activation function instances
    skip: int
        number of layers to skip. Interval for skipping
        ex) skip=2, then add skip layer every other 2 layers

    Example
    -------
    >>> model = Resnet(n_input=10, n_hidden_list=[8,6,5], activation=[nn.Sigmoid(), nn.ReLU(), nn.tanh()], skip=2)
    # activation corresponds to [h1, h2, output]
    >>> print(model)

    '''
    def __init__(self, n_input, n_hidden_list, activation_list, skip=2):
        super(Resnet, self).__init__()
        if type(activation_list) is not list:
            activation_list = [activation_list]*len(n_hidden_list)
        assert len(activation_list)==len(n_hidden_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'
        assert len(n_hidden_list) >= skip, 'number of layers must be equal or greater than skip. len(n_hidden_list): %s, skip: %s'%(len(n_hidden_list), skip)
        assert skip >= 2, 'skip needs to be: skip >= 2, given: %s'%(skip)

        # 1st layer
        layers = [nn.Flatten(1), nn.Linear(n_input, n_hidden_list[0])]
        # Hidden layers ~ Output layer
        layers.extend([nn.Linear(n_hidden_list[i], n_hidden_list[i+1]) for i in range(len(n_hidden_list)-1)])
        self.fc = nn.ModuleList(layers)
        self.activation_list = nn.ModuleList(activation_list)

        # Skip layers
        self.skip = skip
        n_skip_hidden_list = [n_input] + [n_hidden_list[i] for i in range(skip-1, len(n_hidden_list), skip)]
        skip_layers = ([nn.Linear(n_skip_hidden_list[i], n_skip_hidden_list[i+1], bias=False) for i in range(len(n_skip_hidden_list)-1)]) # Bias already present at original layers
        self.skip_layers = nn.ModuleList(skip_layers) # Need ModuleList, not list, so that the model can recognize this layers
        self.n_skip = len(n_hidden_list) // skip # number of skip layers
        # assert self.n_skip == len(self.skip_layers), 'something\'s wrong'

    def forward(self, x):
        last_x = x
        skip_layers = iter(self.skip_layers)
        for i, (layer, activation) in enumerate(zip(self.fc, self.activation_list)):
            if (i+1) % self.skip == 0:
                skip_layer = next(skip_layers)
                x = activation(layer(x)+skip_layer(last_x))
                last_x = x
            else:
                x = activation(layer(x))
        return x
    '''To do:
    - weight initialization? modify?
    - optimize using nn.Sequential() and itertools in forward. compute layers in group
    '''

class DNN_Densenet(nn.Module):
    def __init__(self, n_input, n_hidden_list, activation_list, skip=2):
        super(Densenet, self).__init__()
        if type(activation_list) is not list:
            activation_list = [activation_list]*len(n_hidden_list)
        assert len(activation_list)==len(n_hidden_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'
        assert len(n_hidden_list) >= skip, 'number of layers must be equal or greater than skip. len(n_hidden_list): %s, skip: %s'%(len(n_hidden_list), skip)
        assert skip >= 2, 'skip needs to be: skip >= 2, given: %s'%(skip)

        # 1st layer
        layers = [nn.Linear(n_input, n_hidden_list[0])]
        # Hidden layers ~ Output layer
        layers.extend([nn.Linear(n_hidden_list[i], n_hidden_list[i+1]) for i in range(len(n_hidden_list)-1)])
        self.fc = nn.ModuleList(layers)
        self.activation_list = nn.ModuleList(activation_list)

        # Skip layers
        self.skip = skip
        n_skip_hidden_list = [n_input] + [n_hidden_list[i] for i in range(skip-1, len(n_hidden_list), skip)]
        skip_layers = nn.ModuleDict()
        for i in range(1, len(n_skip_hidden_list)):
            for j in range(i):
                skip_layers[str(j)+'_'+str(i)]=nn.Linear(n_skip_hidden_list[j], n_skip_hidden_list[i], bias=False)
        self.skip_layers = skip_layers

        n = len(n_skip_hidden_list)-1
        self.n_skip = int(n*(n+1)/2)
        assert self.n_skip == len(skip_layers), f'something wrong: n_skip({self.n_skip}), skip_layers({len(skip_layers)})'

    def forward(self, x):
        x_history = [x]
        dest = 1 # Destination index of skip layers
        for i, (layer, activation) in enumerate(zip(self.fc, self.activation_list)):
            if (i+1) % self.skip == 0:
                assert dest == len(x_history), f'Length of x_history{len(x_history)} and destination number{dest} must match'
                skip_x = torch.stack([self.skip_layers[str(src)+'_'+str(dest)](x_) for src, x_ in enumerate(x_history)], dim=-1).sum(dim=-1)
                x = activation(layer(x)+skip_x)
                x_history.append(x)
                dest += 1
            else:
                x = activation(layer(x))

        return x

class CNN(nn.Module):
    def __init__(self, in_channel, n_channel_list, activation_list, kernel_size=3):
        super(CNN, self).__init__()
        if type(activation_list) is not list:
            activation_list = [activation_list]*len(n_channel_list)
        assert len(activation_list)==len(n_channel_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'

        # 1st layer
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_channels=in_channel, out_channels=n_channel_list[0], kernel_size=kernel_size, stride=1, padding=padding), activation_list[0]]
        # Hidden layers ~ Output layer
        for i in range(len(n_channel_list) - 1):
            layers.extend([nn.Conv2d(in_channels=n_channel_list[i], out_channels=n_channel_list[i+1], kernel_size=kernel_size, stride=1, padding=padding), activation_list[i+1]])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CNN_Resnet(nn.Module):
    def __init__(self, in_channel, n_channel_list, activation_list, kernel_size=3, skip=2):
        super().__init__()
        if type(activation_list) is not list:
            activation_list = [activation_list]*len(n_channel_list)
        assert len(activation_list)==len(n_channel_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'
        assert kernel_size % 2 == 1, f'kernel size must be odd number, Got: ({kernel_size})'

        # 1st layer
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_channels=in_channel, out_channels=n_channel_list[0], kernel_size=kernel_size, stride=1, padding=padding)]
        # Hidden layers ~ Output layer
        layers.extend([nn.Conv2d(in_channels=n_channel_list[i], out_channels=n_channel_list[i+1], kernel_size=kernel_size, stride=1, padding=padding) for i in range(len(n_channel_list)-1)])
        self.layers = nn.ModuleList(layers)
        self.activation_list = nn.ModuleList(activation_list)

        # Skip layers
        self.skip = skip
        n_skip_channel_list = [in_channel]+[n_channel_list[i] for i in range(skip-1, len(n_channel_list), skip)]
        skip_layers = [nn.Conv2d(in_channels=n_skip_channel_list[i], out_channels=n_skip_channel_list[i+1], kernel_size=kernel_size, stride=1, padding=padding, bias=False) for i in range(len(n_skip_channel_list)-1)] # Bias already present at original layers
        self.skip_layers = nn.ModuleList(skip_layers) # Need ModuleList, not list, so that the model can recognize this layers
        self.n_skip = len(n_channel_list) // skip # number of skip layers
        # assert self.n_skip == len(self.skip_layers), 'something\'s wrong'

    def forward(self, x):
        last_x = x
        skip_layers = iter(self.skip_layers)
        for i, (layer, activation) in enumerate(zip(self.layers, self.activation_list)):
            if (i+1) % self.skip == 0:
                skip_layer = next(skip_layers)
                x = activation(layer(x)+skip_layer(last_x))
                last_x = x
            else:
                x = activation(layer(x))
        return x

class CNN_Densenet(nn.Module):
    def __init__(self, in_channel, n_channel_list, activation_list, kernel_size=3, skip=2):
        super().__init__()
        if type(activation_list) is not list:
            activation_list = [activation_list]*len(n_channel_list)
        assert len(activation_list)==len(n_channel_list), 'length of layers and activations must match. If you want no activation, use nn.Identity'
        assert kernel_size % 2 == 1, f'kernel size must be odd number, Got: ({kernel_size})'

        # 1st layer
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_channels=in_channel, out_channels=n_channel_list[0], kernel_size=kernel_size, stride=1, padding=padding)]
        # Hidden layers ~ Output layer
        layers.extend([nn.Conv2d(in_channels=n_channel_list[i], out_channels=n_channel_list[i+1], kernel_size=kernel_size, stride=1, padding=padding) for i in range(len(n_channel_list)-1)])
        self.layers = nn.ModuleList(layers)
        self.activation_list = nn.ModuleList(activation_list)

        # Skip layers
        self.skip = skip
        n_skip_channel_list = [in_channel]+[n_channel_list[i] for i in range(skip-1, len(n_channel_list), skip)]

        skip_layers = nn.ModuleDict()
        for i in range(1, len(n_skip_channel_list)):
            for j in range(i):
                skip_layers[str(j)+'_'+str(i)]=nn.Conv2d(in_channels=n_skip_channel_list[j], out_channels=n_skip_channel_list[i], kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.skip_layers = skip_layers

        n = len(n_skip_channel_list)-1
        self.n_skip = int(n*(n+1)/2)
        assert self.n_skip == len(skip_layers), f'something wrong: n_skip({self.n_skip}), skip_layers({len(skip_layers)})'

    def forward(self, x):
        x_history = [x]
        dest = 1 # Destination index of skip layers
        for i, (layer, activation) in enumerate(zip(self.layers, self.activation_list)):
            if (i+1) % self.skip == 0:
                assert dest == len(x_history), f'Length of x_history{len(x_history)} and destination number{dest} must match'
                skip_x = torch.stack([self.skip_layers[str(src)+'_'+str(dest)](x_) for src, x_ in enumerate(x_history)], dim=-1).sum(dim=-1)
                x = activation(layer(x)+skip_x)
                x_history.append(x)
                dest += 1
            else:
                x = activation(layer(x))

        return x

# def residual_computation(sequential, x, connection='sequential'):
#     if connection=='sequential':
#         pass
#     elif connection=='dense':
#         pass
#
#     return

if __name__ == '__main__':
    n_input = 5
    n_hidden_list = [10,20,30,20,10]
    activation_list = [nn.Sigmoid(), nn.ReLU(), nn.LeakyReLU(0.03), nn.Tanh(), nn.Identity()]
    dnn = DNN(n_input, n_hidden_list, activation_list)
    dnn_resnet = DNN_Resnet(n_input, n_hidden_list, activation_list)
    dnn_densenet = DNN_Densenet(n_input, n_hidden_list, activation_list)
    print(dnn)
    print(dnn_resnet)
    print(dnn_densenet)
