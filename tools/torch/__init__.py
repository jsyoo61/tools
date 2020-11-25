
from . import federated_learning, model

def device_of(nn_module):
    return next(nn_module.parameters()).device

def device_of_optimizer(optimizer):
    return optimizer.param_groups[0]['params'][0].device

def to_onehot(x, max_dim):
    '''
    x: list of int

    max_dim: maximum dimension
    '''
    onehot = torch.zeros(len(x),max_dim)
    onehot[range(len(x)), x] = 1
    return onehot
