from .sklearn import *
from .utils import *

if __name__ == '__main__':
    pass

# def add_colorbar(ax):
#     """
#     Add a colorbar to the given axes.

#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes

#     Returns
#     -------
#     C : ndarray of shape (n_classes, n_classes)
#         Explanation variable
#     """
#     from matplotlib import cm
#     from mpl_toolkits.axes_grid1 import make_axes_locatable

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = ax.figure.colorbar(cm.ScalarMappable(norm=ax.images[0].norm, cmap=ax.images[0].cmap), cax=cax)
#     return cbar

def add_colorbar(ax, mappable=None, width=0.02, pad=0.02, divide=False):
    """
    Add a colorbar to the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar added to the axes.
    """
    fig = ax.figure
    if mappable is None:
        if ax.images:
            mappable = ax.images[0]
        else:
            raise ValueError("No mappable found in the axes. Please provide a mappable or ensure the axes contain an image.")

    if divide:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=width, pad=pad)
    else:        
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.x1 + pad, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(mappable, cax=cax)
    # cbar = ax.figure.colorbar(cm.ScalarMappable(norm=ax.images[0].norm, cmap=ax.images[0].cmap), cax=cax)
    # cbar = ax.figure.colorbar(ax.images[0], ax=ax, location='right', pad=0.15)

    return cbar