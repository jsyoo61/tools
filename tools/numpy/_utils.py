import numpy as np

import tools as T

# %%
__all__ = [
'equal',
'merge_dict',
'Squeezer'
]

# %%
def equal(array, axis=None):
    """
    Check if all elements in the array are equal along the specified axis.

    Parameters
    ----------
    array : array-like
        Input array to check for equality.
    axis : int, optional
        Axis along which to check for equality. If None, check the entire array.

    Returns
    -------
    bool or ndarray of bool
        if axis is None, return True if all elements are equal, False otherwise.
        if axis is specified, return an array of bools with axis dimension collapsed 
    """
    array = np.asarray(array)
    if axis is None:
        return np.all(array == array.flat[0])
    else:
        return np.all(array == np.expand_dims(array.take(0, axis=axis), axis=axis), axis=axis)

def merge_dict(ld):
    '''
    :param ld: list of dicts [{}, {}, {}]
    '''
    assert type(ld) == list, f'must give list of dicts, received: {type(ld)}'
    assert T.equal([list(d.keys()) for d in ld]), 'keys for every dict in the list of dicts must be the same'
    keys = ld[0].keys()
    merged_d = {}
    for k in keys:
        lv = [d[k] for d in ld] # list of values
        try:
            merged_d[k] = np.concatenate(lv, axis=0)
        except ValueError:
            merged_d[k] = np.array(lv)
    return merged_d

class Squeezer(object):
    """
    Warning: The class is very unstable, currently used as a temporary adjustment for concatenating / splitting batch dimension.

    Reshape numpy arrays or lists of numpy arrays.
    """
    def __init__(self):
        self.type = None
        self.ndim = None
        self.shape_original = None

    def squeeze(self, x):
        if isinstance(x, np.ndarray):
            self.type = np.ndarray
            self.ndim = x.ndim
            if self.ndim==2:
                self.shape_original = x.shape
            elif x.ndim==3:
                self.shape_original = x.shape
                x.shape = (self.shape_original[0]*self.shape_original[1], self.shape_original[2])
            else:
                raise Exception(f'x.ndim must be 2 or 3, received: {x.ndim}')
        elif isinstance(x, list):
            self.type = list
            if len(x) != 0:
                assert isinstance(x[0], np.ndarray), 'values in list must be np.ndarray'
                assert x[0].ndim==2, f'arrays in list must have ndim==2, received: {x[0].ndim}'
                self.ndim = 3
                self.shape_original = [x_.shape for x_ in x]
                x = np.concatenate(x, axis=0)
        else:
            raise TypeError('Input must be a numpy array or a list of numpy arrays.')
        
        return x

    def unsqueeze(self, x, strict=True):
        if self.type is None or self.ndim is None or self.shape_original is None:
            raise ValueError("Squeezer instance was not initialized correctly.")

        if self.ndim == 3:
            if self.type == np.ndarray:
                if strict:
                    if x.size == np.prod(self.shape_original):
                        x.shape = self.shape_original
                    else:
                        raise ValueError(f'Input shape {x.shape} cannot be reshaped into {self.shape_original}.')
                else:  # Allow flexibility in last dimension
                    expected_elements = np.prod(self.shape_original[:-1])
                    if x.size % expected_elements == 0:
                        x.shape = (*self.shape_original[:-1], -1)
                    else:
                        raise ValueError(f'Cannot reshape {x.shape} flexibly; incorrect number of elements.')
            elif self.type == list:
                split_indices = np.cumsum([shape[0] for shape in self.shape_original[:-1]])
                if x.shape[0] != sum(shape[0] for shape in self.shape_original):
                    raise ValueError(f'Input shape {x.shape} does not match expected concatenated shape.')
                x = np.split(x, split_indices, axis=0)

        return x

# %%
if __name__ == '__main__':
    ld = [{i:i+1 for i in range(5)} for j in range(3)]
    merge_dict(ld)
    ld = [{i:np.arange(i+1) for i in range(5)} for j in range(3)]
    merge_dict(ld)
