import numpy as np
import scipy.stats as st

def ci(std, n, p=0.95):
    z = st.norm.ppf((1+p)/2) # Two-tailed confidence interval
    half_range = z*std/np.sqrt(n)
    return half_range

def interval(distrib, confidence, **kwargs):
    if 'std' in kwargs and 'n' in kwargs:
        scale = kwargs['std']/np.sqrt(kwargs['n'])
        assert 'scale' not in kwargs, 'scale is given but std and n are also given'
        kwargs['scale'] = scale
        del kwargs['std'], kwargs['n']

    return distrib.interval(confidence, **kwargs)

def ci_stats(mean, std, n, p=0.95):
    '''
    calculate confidence interval based on given statistics
    :param mean: 1d array, sample mean
    :param std: 1d array, sample mean
    :param n: 1d array, sample size
    :param p: 1d array, confidence level
    '''
    # z = st.norm.ppf(p)
    # half_range = z*std/n
    half_range = ci(std=std, n=n)

    confidence_interval = {
    'low': mean-half_range,
    'high': mean+half_range
    }
    return confidence_interval

def ci_samples(x, p=0.95):
    '''
    calculate confidence interval based on samples
    '''
    mean = np.mean(x)
    std = np.std(x)
    n = len(x)
    return ci_stats(mean, std, n, p=p)
