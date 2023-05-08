
import matplotlib.pyplot as plt
import numpy as np
from . import numpy as tnp

class ValueTracker(object):
    """ ValueTracker."""

    def __repr__(self):
        return f'<ValueTracker>\nx: {self.x}\ny: {self.y}'

    def __init__(self):
        self.reset()

    def __len__(self):
        return len(self.y)

    def __iadd__(self, other):
        self.x.extend(other.x)
        self.y.extend(other.y)
        self.label.extend(other.label)
        self.n_step += len(other.x)
        return self

    def __add__(self, other):
        self = deepcopy(self)
        self.x.extend(other.x)
        self.y.extend(other.y)
        self.label.extend(other.label)
        self.n_step += len(other.x)
        return self

    def reset(self):
        self.x = []
        self.y = []
        self.label = []
        self.n_step = 0

    def numpy(self):
        return np.array(self.x), np.array(self.y), np.array(self.label)

    def step(self, x, y, label=None):
        if hasattr(x, '__len__'):
            assert hasattr(y, '__len__')
            assert len(x)==len(y)
            self.x.extend(x)
            self.y.extend(y)
            if label != None:
                assert len(y)==len(label)
                self.label.extend(label)
            self.n_step += len(x)

        else:
            self.x.append(x)
            self.y.append(y)
            if label != None:
                self.label.append(label)
            self.n_step += 1

    def plot(self, w=9, color='tab:blue', ax=None):
        x = np.array(self.x)
        y = np.array(self.y)
        y_smooth = tnp.moving_mean(y, w)
        if ax==None:
            ax = plt.gca()
        ax.plot(x, y, color=color, alpha=0.4)
        ax.plot(x, y_smooth, color=color)
        return ax

    def mean(self):
        return np.mean(self.y)
    def min(self):
        return np.min(self.y)
    def max(self):
        return np.max(self.y)

class AverageMeter(object):
    """Computes and stores the average and current value
    Variables
    ---------
    self.val
    self.avg
    self.sum
    self.count
    """
    # TODO: maybe keep track of each values? or just merge with valuetracker?

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def step(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DictList(object):
    """Dictionary of lists"""
    def __init__(self, keys):
        self._dict = {}

    def append(self, data):
        assert type(data) in [list, dict]
        if type(data)==dict:
            assert set(self._dict.keys())==set(data.keys()), f'allowed keys are: {self._dict.keys()}, received: {data.keys()}'
            for key, value in data.items():
                self._dict[key].append(item)
        else:
            warnings.warn('Appending with list is not recommended as this cannot ensure the data are being appended to the right place.')
            for key, value in zip(self._dict.items(), data):
                self._dict[key].append(value)
