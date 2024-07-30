import numbers

import numpy as np
import pandas as pd

def isclose(a, b, rtol=1e-09, atol=0.0, equal_nan=False):
    """
    Wraparound np.isclose for pandas.DataFrame

    Parameters
    ----------
    a, b : pandas.DataFrame
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    C : pandas.DataFrame 
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.
    """
    assert a.shape == b.shape, "Shape mismatch"
    assert (a.columns == b.columns).all(), "Column mismatch"
    assert (a.index == b.index).all(), "Index mismatch"

    close_list = []
    for col in a.columns:
        if issubclass(a[col].dtype.type, numbers.Number):
            close = np.isclose(a[col], b[col], rtol=rtol, atol=atol, equal_nan=equal_nan)
        else:
            close = (a[col] == b[col]).to_numpy()
        close_list.append(close)
    close = np.stack(close_list, axis=1)

    return pd.DataFrame(close, columns=a.columns, index=a.index)

if __name__ == '__main__':
    pass
