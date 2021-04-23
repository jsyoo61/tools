
from . import *

def standardize(array, axis=None, ep=1e-20):
    return (array - array.mean(axis=axis))/(array.std(axis=axis)+ep)

def equal(array):
    if len(array)<=1:
        return True
    else:
        return (array[0]==array[1:]).all()
