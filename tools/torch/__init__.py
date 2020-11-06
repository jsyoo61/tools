
from . import aggregation, model

def device_of(nn_module):
    return next(nn_module.parameters()).device

def device_of_optimizer(optimizer):
    return optimizer.param_groups[0]['params'][0].device
