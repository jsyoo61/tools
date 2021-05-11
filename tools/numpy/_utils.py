import numpy as np

import tools as T

# %%
__all__ = [
'equal',
'merge_dict',
]

# %%
def equal(array):
    if len(array)<=1:
        return True
    else:
        return (array[0]==array[1:]).all()

def merge_dict(ld):
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
    
# %%
if __name__ == '__main__':
    ld = [{i:i+1 for i in range(5)} for j in range(3)]
    merge_dict(ld)
    ld = [{i:np.arange(i+1) for i in range(5)} for j in range(3)]
    merge_dict(ld)
