
from . import federated_learning, model
from ._pandas import *
from .utils import *
from . import data

def to_onehot(x, max_dim):
    '''
    x: list of int

    max_dim: maximum dimension
    '''
    onehot = torch.zeros(len(x),max_dim)
    onehot[range(len(x)), x] = 1
    return onehot
