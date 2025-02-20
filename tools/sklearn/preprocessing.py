from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def standardize(x, scaler=None):
    """
    Standardize (z-score/gaussian normalization) x without having to make a StandardScaler() object.

    Parameters
    ----------
    x : ndarray of shape (n_observation, n_channel) 
        data to be standardized.
    scaler: sklearn.preprocessing.StandardScaler(), default=None
        scaler to use (that has already fit())

    Returns
    -------
    x_standardized : ndarray of shape (n_observation, n_channel)
        Normalized data
    scaler : sklearn.preprocessing.StandardScaler()
        Scaler used for normalization
    """
    if scaler is None:
        scaler = StandardScaler()
        x_standardized = scaler.fit_transform(x)
    else:
        x_standardized = scaler.transform(x)
    
    return x_standardized, scaler

def project_pca(x, var_threshold=None, n_pc=None, solver=None, random_state=0):
    """
    project x onto PC space without having to make a PCA() object.

    Parameters
    ----------
    x : ndarray of shape (n_observation, n_channel)
        data to be projected.
    solver: sklearn.decomposition.PCA(), default=None
        solver to use (that has already fit())

    Returns
    -------
    x_projected : ndarray of shape (n_observation, n_channel)
        Normalized data
    solver : sklearn.decomposition.PCA()
        Solver used for PCA projection
    """
    assert (var_threshold is None) ^ (n_pc is None), 'Only one of var_threshold or n_pc must be specified'

    # If svd_solver is not full, result is random
    if solver is None:
        solver = PCA(svd_solver='full', random_state=random_state)
        x_pc = solver.fit_transform(x)
    else:
        x_pc = solver.transform(x)

    if var_threshold is not None:
        n_pc = np.argmax(np.cumsum(solver.explained_variance_ratio_) > var_threshold) + 1
    elif n_pc is not None:
        pass
    x_pc = x_pc[:,:n_pc]

    return x_pc, solver

if __name__=='__main__':
    import tools as T

    x = np.random.rand(2,3,5)
    print(x.shape)
    squeezer = T.numpy.Squeezer()
    x = squeezer.squeeze(x)
    print(x.shape)
    x_s, scaler = T.sklearn.preprocessing.standardize(x)
    x_s = squeezer.unsqueeze(x_s)
    print(x_s.shape)

    # %%
    x = np.random.rand(2,3,5)
    print(x.shape)
    squeezer = T.numpy.Squeezer()
    x = squeezer.squeeze(x)
    print(x.shape)
    x_pc, solver = T.sklearn.preprocessing.project_pca(x, n_pc=3)
    x_pc = squeezer.unsqueeze(x_pc, strict=False)
    print(x_pc.shape)
