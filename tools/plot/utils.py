# import logging
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .. import numpy as tnp
# log = logging.getLogger(__name__)

__all__ = [
'get_xtick_seconds',
'multisave',
'squaresubplots',
]

# %%
def get_xtick_seconds(ax, sr):
    '''
    :param ax: axis instance
    :param sec: second which maximum length indicates. it is assumed the xtick starts at 0
    '''
    # xticks = np.linspace(0,sec,len(ax_real.get_xticklabels())-2)
    # xticks = np.round(xticks, 3)
    # d = xticks[1]-xticks[0]
    # xticks = np.concatenate([[xticks[0] - d],xticks,[xticks[-1]+d]])
    # return xticks
    return ax.get_xticks() / sr

def multisave(fig, path, dpi=300):
    '''
    save given figure in .png, .eps, .svg format
    :param fig: pyplot figure instance
    :param path: path of the figure without extension (.extension)
    '''
    if '.' in os.path.basename(path):
        warnings.warn("Do not specify '.' in name. assuming it's part of the filename...")

        # log.warning('Do not specify extension in multisave(). Removing extension...')
        # path = os.path.splitext(path)[0]

    fig.savefig(path+'.png', dpi=dpi)
    fig.savefig(path+'.tiff', dpi=dpi)
    fig.savefig(path+'.eps')
    fig.savefig(path+'.svg')

def plot_values(valuetracker_list, ax=None):
    n_line = len(valuetracker_list)
    if ax is None:
        fig, ax = plt.subplots()
    color_list = it.cycle(mcolors.TABLEAU_COLORS)
    for valuetracker, color in zip(valuetracker_list, color_list):
        y_smooth = tnp.moving_mean(valuetracker.y, 9)
        ax.plot(valuetracker.x, valuetracker.y, color=color, alpha=0.4)
        ax.plot(valuetracker.x, y_smooth, color=color)
    return ax

def plot_trainval(valuetracker_list, ax=None):
    n_line = len(valuetracker_list)
    if ax is None:
        fig, ax = plt.subplots()
    color_list = ['tab:blue', 'tab:orange']
    line_list = ['-', 'x-']
    labels = ['train', 'val']
    for valuetracker, color, line, label in zip(valuetracker_list, color_list, line_list, labels):
        y_smooth = tnp.moving_mean(valuetracker.y, 9)
        ax.plot(valuetracker.x, valuetracker.y, color=color, alpha=0.4, label=label)
        ax.plot(valuetracker.x, y_smooth, line, color=color, label=label+'_smooth')
    return ax

def squaresubplots(n_axes, figsize=None, axsize=None):
    """
    Return the axes and figures of a square grid of subplots.

    Parameters
    ----------
    n_axes : number of axes (int)
    figsize : size of the figure, (tuple, optional)
    
    Returns
    -------
    fig : figure
    axes : np.ndarray of shape (n_rows, n_cols)
    """
    n = np.sqrt(n_axes)
    if n.is_integer():
        nrows, ncols = int(n), int(n)
    else:
        nrows, ncols = int(n)+1, int(n)
        if nrows*ncols < n_axes:
            ncols += 1

    if axsize is not None:
        figsize = (axsize[0]*ncols, axsize[1]*nrows)
        if figsize is not None:
            warnings.warn('axsize is provided, figsize will be ignored.')

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if not issubclass(type(axes), np.ndarray):
        axes = np.array([[axes]])
    return fig, axes

if __name__ == '__main__':
    pass
