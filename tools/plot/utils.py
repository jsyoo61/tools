# import logging
import os
import warnings

import numpy as np

# log = logging.getLogger(__name__)

__all__ = [
'get_xtick_seconds',
'multisave',
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

def multisave(fig, path):
    '''
    save given figure in .png, .eps, .svg format
    :param fig: pyplot figure instance
    :param path: path of the figure without extension (.extension)
    '''
    if '.' in os.path.basename(path):
        warnings.warn("Do not specify '.' in name. assuming it's part of the filename...")

        # log.warning('Do not specify extension in multisave(). Removing extension...')
        # path = os.path.splitext(path)[0]

    fig.savefig(path+'.png')
    fig.savefig(path+'.eps')
    fig.savefig(path+'.svg')

if __name__ == '__main__':
    pass
