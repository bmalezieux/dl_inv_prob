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
        X = (np.random.random((n_dicos, n_components, N)) > (1-s)).astype(float)
        X *= np.random.normal(scale=1, size=(n_dicos, n_components, N))
    else:
        X = (rng.random((n_dicos, n_components, N)) > (1-s)).astype(float)
        X *= rng.normal(scale=1, size=(n_dicos, n_components, N))
    return D @ X, X


def recovery_score(D, Dref):
    """
    Comparison between two dictionaries
    """
    cost_matrix = np.abs(Dref.T @ D)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    score = cost_matrix[row_ind, col_ind].sum() / D.shape[1]

    return score


def extract_patches(img, dim_patch):
    """Extract non overlapping square patches from an image in a grid pattern.

    Parameters
    ----------
    img : numpy.array, shape (dim_img, dim_img)
        Square image with a single channel
    dim_patch : int
        Dimension of the square patches to extract

    Returns
    -------
    patches : numpy.array, shape (nb_patches, dim_patch, dim_patch)
        3d array containg the extracted patches
    """
    dim_img = img.shape[0]
    nb_patches = (dim_img // dim_patch) ** 2
    res = np.zeros((nb_patches, dim_patch, dim_patch))
    i_patch = 0
    for i in range(0, dim_img, dim_patch):
        for j in range(0, dim_img, dim_patch):
            res[i_patch, :, :] = img[i:i + dim_patch, j:j + dim_patch]
            i_patch += 1
    patches = np.array(res)
    
    return patches


def combine_patches(patches):
    """Combine extracted patches (in a grid fashion) into the original image.

    Parameters
    ----------
    patches : numpy.array, shape (nb_patches, dim_patch, dim_patch)
        3d array containing the extracted patches

    Returns
    -------
    img : numpy.array, shape (dim_img, dim_img)
        original image
    """
    n_patches = patches.shape[0]
    dim_grid = int(np.sqrt(n_patches))

    rows = [np.hstack([patches[i] for i in range(k, k + dim_grid)])
            for k in range(0, n_patches, dim_grid)]
    img = np.vstack(rows)
    
    return img


def psnr(rec, ref):
    """Compute the peak signal-to-noise ratio for images on 8 bits.

    Parameters
    ----------
    rec : numpy.array, shape (height, width)
        reconstructed image
    ref : numpy.array, shape (height, width)
        original image

    Returns
    -------
    psnr : float
        psnr of the reconstructed image
    """
    mse = np.square(rec - ref).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    
    return psnr
