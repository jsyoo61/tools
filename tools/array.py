import numpy as np
import torch
import tools as T


# TODO: change name mangic_cat() -> concat()
__all__ = [
'magic_cat',
]
# %%
def magic_cat(datas, axis=0, out=None):
    '''concatenate numpy or torch tensors'''
    assert T.equal([type(data) for data in datas]), 'Given data in datas must be the same type'
    datatype = type(datas[0])
    if datatype == torch.Tensor:
        return torch.cat(datas, dim=axis, out=out)
    elif datatype == np.ndarray or list:
        return np.concatenate(datas, axis=axis, out=out)
    else:
        raise Exception(f'Elements in data must be either [list, np.ndarray, torch.Tensor], received: {datatype}')
