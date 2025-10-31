import numpy as np

import tools as T
import warnings

# %%
__all__ = [
'equal',
'isclose',
'merge_dict',
'Squeezer',
'unique_isclose'
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

def isclose(array, axis, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Same with numpy.isclose, but casted on a single array not multiple.

    Parameters
    ----------
    array : array-like
    axis : int or tuple of ints
        Axis to check for isclose
    rtol : array-like
        The relative tolerance parameter
    atol: array-like
        The absolute tolerance parameter
    equal_nan : bool
        Whether to compare NaN’s as equal. If True, NaN’s in a will be considered equal to NaN’s in b in the output array.
        
    Returns
    -------
    array : boolean ndarray of shape without the specified axis

    See Also
    --------
    np.isclose (https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
    """
    if not T.is_iterable(axis):
        axis = (axis,)

    axis = tuple(array.ndim+ax_ if ax_<0 else ax_ for ax_ in axis)
    index_ref = tuple(slice(None) if axis_ not in axis else 0 for axis_ in range(array.ndim))

    array_ref = np.expand_dims(array[index_ref], axis=axis)

    return np.isclose(array, array_ref, rtol=rtol, atol=atol, equal_nan=equal_nan).all(axis) # True only if isclose across all elements within the specified axis

def unique_isclose(ar, rtol=1e-05, atol=1e-08, return_index=False, return_inverse=False, return_counts=False, equal_nan=True, sorted=True):
    """
    Combination of np.unique and np.isclose. Basically np.unique with numerical evaluation.

    Parameters
    ----------
    ar : 1-d array

    rtol : array-like
        The relative tolerance parameter
    atol: array-like
        The absolute tolerance parameter
    equal_nan : bool, optional
        Whether to compare NaN’s as equal. If True, NaN’s in a will be considered equal to NaN’s in b in the output array.

    return_index : bool, optional (NotImplemented)
        If True, also return the indices of ar (along the specified axis, if provided, or in the flattened array) that result in the unique array.
    return_inverse: bool, optional (NotImplemented)
        If True, also return the indices of the unique array (for the specified axis, if provided) that can be used to reconstruct ar.
    return_counts: bool, optional
        If True, also return the number of times each unique item appears in ar.

    Returns
    -------
    unique : ndarray

    unique_counts: ndarray, optional

    See Also
    --------
    np.unique (https://numpy.org/devdocs/reference/generated/numpy.unique.html)
    np.isclose (https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
    """

    assert ar.ndim==1, 'Only 1-dimensional arrays are supported'

    if return_index:
        warnings.warn("return_index NotImplemented")

    if return_inverse:
        warnings.warn("return_inverse NotImplemented")
    
    if not sorted:
        warnings.warn("Currently the unique elements are always sorted.")

    ar = np.sort(ar)

    unique_i = ~np.isclose(ar[1:], ar[:-1], rtol=rtol, atol=atol, equal_nan=equal_nan)
    unique_i = np.concatenate([[True], unique_i])
    unique = ar[unique_i]

    if return_counts:
        counts = np.where(unique_i)[0]
        counts = np.concatenate([counts, [ar.size]])
        counts = np.diff(counts)

        return unique, counts

    return unique

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
