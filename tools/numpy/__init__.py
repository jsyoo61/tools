from ._f import *
from ._utils import *

if __name__ == '__main__':
    ld = [{i:i+1 for i in range(5)} for j in range(3)]
    merge_dict(ld)
    ld = [{i:np.arange(i+1) for i in range(5)} for j in range(3)]
    merge_dict(ld)
