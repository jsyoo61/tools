import numpy as np
import scipy.stats as st

# from . import model_selection
def ci(mean, std, n, p=0.95):
    '''
    calculate confidence interval based on given statistics
    :param mean: 1d array, sample mean
    :param std: 1d array, sample mean
    :param n: 1d array, sample size
    :param p: 1d array, confidence level
    '''
    z = st.norm.ppf(p)
    half_range = z*std/n

    confidence_interval = {
    'low': mean-half_range,
    'high': mean+half_range
    }
    return confidence_interval

def ci():
    '''
    calculate confidence interval based on samples
    '''


    return
