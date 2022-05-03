import numpy as np

from scipy.optimize import linear_sum_assignment


def generate_dico(n_components, dim_signal, rng=None):
    """
    Generates a normalized random gaussian dictionary

    Parameters
    ----------
    n_components : int
        Number of atoms
    dim_signal : int
        Dimension of the signal
    rng : np.random.Generator

    Returns
    -------
    np.array (dim_signal, n_components)
        Dictionary
    """
    if rng is None:
        D = np.random.normal(size=(dim_signal, n_components))
    else:
        D = rng.normal(size=(dim_signal, n_components))
    D /= np.sqrt(np.sum(D**2, axis=0))
    return D


def generate_data(D, N, s=0.1, rng=None):
    """
    Generate data from dictionaries

    Parameters
    ----------
    D : np.array (n_dicos, dim_signal, n_components)
        dictionary
    N : int
        number of samples per dictionary
    s : float, optional
        sparsity, by default 0.3
    rng : np.random.Generator

    Returns
    -------
    np.array (n_dicos, dim_signal, N), np.array (n_dicos, n_components, N)
        signal, sparse codes
    """
    n_components = D.shape[2]
    n_dicos = D.shape[0]
    if rng is None:
        X = (rng.random((n_dicos, n_components, N)) > (1-s)).astype(float)
        X *= rng.normal(scale=1, size=(n_dicos, n_components, N))
    else:
        X = (rng.random((n_dicos, n_components, N)) > (1-s)).astype(float)
        X *= rng.normal(scale=1, size=(n_dicos, n_components, N))
    return D @ X, X


def recovery_score(D, Dref):
    """
    Comparison between two dictionaries
    """
    cost_matrix = np.abs(Dref.T@D)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    score = cost_matrix[row_ind, col_ind].sum() / D.shape[1]

    return score
