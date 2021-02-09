import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D


def extract_feature(model, input_shape):
    '''
    Reverse-optimizing input to get feature that maximizes the unit
    "Understanding deep image representations by inverting them"
    '''
    x = torch.rand(*input_shape, requires_grad=True)
    optimizer = optim.Adam()



# %%
if __name__ == '__main__':

    from tools.torch.model import DNN
    import numpy as np
    x = torch.tensor([0,0,0,1,1,0,1,1], dtype=torch.float).reshape(-1,2)
    y = torch.logical_xor(x[:,0], x[:,1]).unsqueeze(-1).float()

    input_shape =(2,)
    model = DNN(n_input=np.prod(input_shape), n_hidden_list=[20,1], activation_list=nn.Sigmoid())

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for i in range(300):
        y_hat = model(x)
        loss = F.binary_cross_entropy(y_hat, y)
        # loss = F.l1_loss(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        print(y, model(x))

    # %%
    x = torch.rand(*input_shape).unsqueeze(0)
    x.requires_grad_(True)

    optimizer = optim.Adam([x], lr=1e-2)
    model.eval()
    model.requires_grad_(False)
    optimizer.param_groups

    for i in range(300):
        y = -model(torch.clamp(x, min=0, max=1))
        optimizer.zero_grad()
        y.backward()
        optimizer.step()

    torch.clamp(x, min=0, max=1)
    model(x)
    model.requires_grad_(True)
