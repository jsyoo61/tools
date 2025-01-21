import random
import multiprocessing

import numpy as np
import torch

__all__ = [
'get_device',
'to',
'multiprocessing_device',
'nanparam',
'nangrad',
'param_same',
'print_rng_state',
'seed',
]

def get_device(network):
    return next(network.parameters()).device

# TODO: Merge into get_device
def device_of_optimizer(optimizer):
    return optimizer.param_groups[0]['params'][0].device

def to(data, *args, **kwargs):
    if isinstance(data, (list, tuple)):
        return tuple(getattr(tensor, 'to')(*args, **kwargs) for tensor in data)
    elif isinstance(data, dict):
        return {k: getattr(tensor, 'to')(*args, **kwargs) for k, tensor in data.items()}
    else:
        return data.to(*args, **kwargs)

def spread_device(gpu_id = None):
    '''
    what's this for?
    '''
    if gpu_id != None:
        gpu_id = gpu_id % torch.cuda.device_count()
    else:
        gpu_id = torch.cuda.default_stream().device.index
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return device

def process_number():
    p=multiprocessing.current_process()
    try:
        worker_num = int(p.name.split('-')[-1]) # Fails if it's not integer
        return worker_num
    except ValueError:
        return -1

def multiprocessing_device(gpu_id = None):
    '''
    device setting for hydra multiprocessing
    '''
    # CPU
    if gpu_id == -1 or not torch.cuda.is_available():
        device = torch.device('cpu')
    # GPU
    else:
        if gpu_id == None:
            # use distributed device, or default device
            p=multiprocessing.current_process()
            #
            try:
                worker_num = int(p.name.split('-')[-1]) # Fails if it's not integer
                gpu_id = worker_num % torch.cuda.device_count()
                device = torch.device(f"cuda:{gpu_id}")
            except ValueError: # Parent process
                device = torch.cuda.default_stream().device
        else:
            device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    # log.info(f'device: {device}') # integrate logging?

    return device

def nanparam(model):
    for p in model.parameters():
        if torch.isnan(p).any():
            return True
    return False

def nangrad(model):
    for p in model.parameters():
        if torch.isnan(p.grad).any():
            return True
    return False

def param_same(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        same = (p1==p2).all()
        if not same:
            return False
    return True

def print_rng_state(n=10, nonzero=True):
    rngs = torch.random.get_rng_state().numpy().tolist()
    if nonzero:
        rngs = [i for i in rngs if i != 0]
    print(rngs[:n])


def seed(random_seed, strict=False):
    '''

    '''
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if strict:
        # Following is verbose, but just in case.
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # deterministic cnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
