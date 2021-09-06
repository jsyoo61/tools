import numpy as np

# %%
__all__ = [
'standardize',
'binarize',
'moving_mean',
'angle',
]

# %%
def standardize(array, axis=None, ep=1e-20):
    return (array - array.mean(axis=axis))/(array.std(axis=axis)+ep)

def binarize(array, threshold):
    result = np.zeros_like(array)
    result[array>=threshold] = 1
    return result

def moving_mean(x, w):
    odd = bool(w%2)
    edge = w//2
    if odd:
        x = np.pad(x, w//2+1, mode='edge')
        x = np.cumsum(x).astype(np.float64)
        x = (x[w:] - x[:-w])/w
        return x[:-1]
    else:
        x = np.pad(x, w//2, mode='edge')
        x = np.cumsum(x).astype(np.float64)
        x = (x[w:] - x[:-w])/w
        return x

def angle(x1, x2):
    '''angle between two vectors, derived from cosine rule
    return theta within range of [0,np.pi]'''
    theta = np.arccos(x1@x2/(np.linalg.norm(x1)*np.linalg.norm(x2)))
    return theta

def sigmoid(x):
    return 1/(1+np.exp(-x))

# %%
if __name__ == '__main__':
    pass

    import numpy as np
    x1=np.array([0,0,1])
    x2=np.array([1,0,0])
