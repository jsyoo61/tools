import numpy as np
import utils as U

# %%
def add_batch_dim(data):
    """
    adds batch dimension (axis=0)

    Parameters
    ----------
    data : numpy.ndarray or dict of arrays

    Returns
    -------
    data : numpy.ndarray or dict of arrays with axis=0 dimension added
    """
    if isinstance(data, dict):
        data = {k: v[None,...] for k, v in data.items()}
    else:
        data = data[None,...]
    return data

def remove_batch_dim(data):
    if isinstance(data, dict):
        data = {k: v.squeeze(0) for k, v in data.items()}
    else:
        data = data.squeeze(0)
    return data

def __eq__(data1, data2):
    if isinstance(data1, dict) and isinstance(data2, dict):
        k1, k2 = sorted(list(data1.keys())), sorted(list(data2.keys()))
        if k1!=k2:
            return False
        else:
            equal_values = [all(data1[k]==data2[k]) for k in k1]
            return all(equal_values)
    else:
        return data1==data2

def __getitem__(data, idx):
    if isinstance(data, dict):
        return {k: v[idx] for k, v in data.items()}
    else:
        return data[idx]

def __iter__(data):
    if isinstance(data, dict):
        iterators = {k: iter(v) for k, v in data.items()}
        while True:
            try:
                yield {k: next(v) for k, v in iterators.items()}
            except StopIteration:
                break
    else:
        return iter(data)

def __len__(data):
    if isinstance(data, dict):
        l_len = [len(v) for v in data.values()]
        assert U.equal(l_len)
        return l_len[0]

    else:
        return len(data)

def array(data, dtype=None):
    '''
    data: iterable or iterable of dicts
    '''
    if isinstance(data, dict):
        return {k: np.array(v, dtype=dtype) for k, v in data.items()}
    else:
        np.array(data, dtype=dtype)

def astype(data, dtype):
    if isinstance(data, dict):
        return {k: v.astype(dtype) for k, v in data.items()}
    else:
        data.astype(dtype)

def broadcast_to(data, shape):
    if isinstance(data, dict):
        return {k: np.broadcast_to(v, shape=shape) for k, v in data.items()}
    else:
        return np.broadcast_to(data, shape=shape)

def shape(data):
    if isinstance(data, dict):
        l_shape = [v.shape for v in data.values()]
        assert U.equal(l_shape), "shape within dict are not equal."
        return l_shape[0]
    else:
        return data.shape

def stack(data, axis):
    if isinstance(data, dict):
        return {k: np.stack(v, axis=axis) for k, v in data.items()}
    else:
        return np.stack(data, axis=axis)

def nwise(data, n):
    """
    Returns n-elements from the data sequentially.

    Returns
    -------
    generator object

    See Also
    --------
    itertools.pairwise : Same as nwise but with n=2

    Examples
    --------
    >>> list(nwise(np.arange(5), n=3))
    [array([0, 1, 2]), array([1, 2, 3]), array([2, 3, 4])]
    """
    if isinstance(data, dict):
        l_shape = [v.shape for v in data.values()]
        assert U.equal(l_shape), "shape within dict are not equal."
        data_shape = l_shape[0]
        for i in range(data_shape[0]-n+1):
            yield {k: v[i:i+n] for k, v in data.items()}
    else:
        for i in range(len(data)-n+1):
            yield data[i:i+n]

def concatenate(data, axis):
    if isinstance(data, dict):
        return {k: np.concatenate(v, axis=axis) for k, v in data.items()}
    else:
        return np.concatenate(data, axis=axis)

def to(data, **kwargs):
    if isinstance(data, dict):
        return {k: v.to(**kwargs) for k, v in data.items()}
    else:
        return data.to(**kwargs)
