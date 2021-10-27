import random

import numpy as np
import torch

# %%
def choice(a, size=None, replace=True):
    '''
    Randomly choose elements from given iterable.

    Parameters
    ----------
    a: int or list
        choice of

    '''
    m = len(a)
    # Single sample
    if size==None:
        random_i = np.random.randint(m)
        return a[random_i]
    else:
        num_sample = np.prod(size)

    # Multi-Sample
    a=np.asarray(a)
    if replace==True:
        random_i = np.random.randint(m, size = num_sample)
        return a[random_i].reshape(size)
    else:
        assert m >= num_sample, 'entries of array cannot exceed number of samples'
        random_i = np.arange(m)
        np.random.shuffle(random_i)
        random_i = random_i[:num_sample]
        return a[random_i].reshape(size)

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
