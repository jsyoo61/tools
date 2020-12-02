
from . import *

def standardize(array, axis=None, ep=1e-20):
    return (array - array.mean(axis=axis))/(array.std(axis=axis)+ep)
