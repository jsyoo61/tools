import numpy as np

# %%
__all__ = [
'standardize',
'moving_mean',
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

# %%
if __name__ == '__main__':
    pass
