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


# Just use merge_dict and [value for value in variable]

# def dict_np_cat(d_array, axis=0):
#     '''
#     concatenates list of numpy array and returns concatenated dict of numpy arrays
#     '''
#     d_cat = {}
#     for key in d_array.keys():
#         d[key] = d
#     np.concatenate
#     return d_cat
#
# def dict_torchcat(d_array, axis=0):
#     pass
#
# def dict_magic_cat(d_array, axis=0):
#     pass
