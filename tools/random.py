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
