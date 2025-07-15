from .sklearn import *
from .utils import *

if __name__ == '__main__':
    pass

def add_colorbar(ax):
    """
    Add a colorbar to the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Explanation variable
    """
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm=ax.images[0].norm, cmap=ax.images[0].cmap), cax=cax)
    return cbar