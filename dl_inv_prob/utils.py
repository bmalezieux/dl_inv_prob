import hashlib
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate2d
from sklearn.datasets import load_digits
import torch
import torchvision.transforms as T


def generate_dico(n_components, dim_signal, rng=None):
    """
    Generate a normalized random gaussian dictionary.

    Parameters
    ----------
    n_components : int
        Number of atoms
    dim_signal : int
        Dimension of the signal
    rng : np.random.Generator, optional (default None)
        Random generator

    Returns
    -------
    ndarray, shape (dim_signal, n_components)
        Generated dictionary
    """
    if rng is None:
        D = np.random.normal(size=(dim_signal, n_components))
    else:
        D = rng.normal(size=(dim_signal, n_components))
    D /= np.sqrt(np.sum(D**2, axis=0))
    return D


def generate_data(D, N, s=0.1, rng=None):
    """
    Generate data from dictionaries.

    Parameters
    ----------
    D : ndarray, shape (n_dicos, dim_signal, n_components)
        Dictionary
    N : int
        number of samples per dictionary
    s : float, optional (default 0.3)
        Sparsity
    rng : numpy.random.Generator, optional (default None)
        Random generator

    Returns
    -------
    signal : ndarray, shape (n_dicos, dim_signal, N)
        Generated signal
    X : ndarray, shape (n_dicos, n_components, N)
        Generated sparse codes
    """
    n_components = D.shape[2]
    n_dicos = D.shape[0]
    if rng is None:
        X = np.random.random((n_dicos, n_components, N)) > (1 - s)
        X = X.astype(float)
        X *= np.random.normal(scale=1, size=(n_dicos, n_components, N))
    else:
        X = (rng.random((n_dicos, n_components, N)) > (1 - s)).astype(float)
        X *= rng.normal(scale=1, size=(n_dicos, n_components, N))
    signal = D @ X

    return signal, X


def recovery_score(D, D_ref):
    """
    Compute a similarity score in [0, 1] between two dictionaries.

    Parameters
    ----------
    D : ndarray, shape (dim_signal, n_components)
        Dictionary
    Dref : ndarray, shape (dim_signal, n_components)
        Reference dictionary

    Returns
    -------
    score : float
        _Recovery score in [0, 1]
    """
    cost_matrix = np.abs(D_ref.T @ D)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    score = cost_matrix[row_ind, col_ind].sum() / D.shape[1]

    return score


def conv_recovery_score(D, D_ref):
    """Compute a similarity score between convolutional dictionaries.

    Parameters
    ----------
    D : ndarray, shape (n_atoms, atom_height, atom_width)
        Dictionary
    D_ref : ndarray, shape (n_atoms, atom_height, atom_width)
        Reference dictionary

    Returns
    -------
    score : float
        Similarity score in [0, 1] between D and D_ref
    """
    corr = np.zeros((D.shape[0], D_ref.shape[0]))

    for i in range(D.shape[0]):
        for j in range(D_ref.shape[0]):
            corr_fun = correlate2d(D[i, :, :], D_ref[j, :, :])
            corr[i, j] = np.max(np.abs(corr_fun))

    row_index, col_index = linear_sum_assignment(corr, maximize=True)
    score = corr[row_index, col_index].mean()

    return score


def extract_patches(img, dim_patch):
    """
    Extract a grid of non-overlapping square patches from a square image.

    Parameters
    ----------
    img : ndarray, shape (dim_img, dim_img)
        Square image with a single channel
    dim_patch : int
        Dimension of the square patches to extract

    Returns
    -------
    patches : ndarray, shape (nb_patches, dim_patch, dim_patch)
        3d array containg the extracted patches
    """
    dim_img = img.shape[0]
    nb_patches = (dim_img // dim_patch) ** 2
    res = np.zeros((nb_patches, dim_patch, dim_patch))
    i_patch = 0
    for i in range(0, dim_img, dim_patch):
        for j in range(0, dim_img, dim_patch):
            res[i_patch, :, :] = img[i : i + dim_patch, j : j + dim_patch]
            i_patch += 1
    patches = np.array(res)

    return patches


def combine_patches(patches):
    """
    Combine extracted patches (in a grid fashion) into the original image.

    Parameters
    ----------
    patches : ndarray, shape (nb_patches, dim_patch, dim_patch)
        3d array containing the extracted patches

    Returns
    -------
    img : ndarray, shape (dim_img, dim_img)
        original image
    """
    n_patches = patches.shape[0]
    dim_grid = int(np.sqrt(n_patches))

    rows = [
        np.hstack([patches[i] for i in range(k, k + dim_grid)])
        for k in range(0, n_patches, dim_grid)
    ]
    img = np.vstack(rows)

    return img


def psnr(rec, ref):
    """
    Compute the peak signal-to-noise ratio for grey images in [0, 1].

    Parameters
    ----------
    rec : ndarray, shape (height, width)
        reconstructed image
    ref : ndarray, shape (height, width)
        original image

    Returns
    -------
    psnr : float
        psnr of the reconstructed image
    """
    mse = np.square(rec - ref).mean()
    psnr = 10 * np.log10(1 / mse)

    return psnr


def create_patches_overlap(im, m, A=None):
    """
    Create an array of flattened overlapping patches.

    Parameters
    ----------
    im : ndarray, shape (height, width)
        Input image
    m : int
        Patch dimension
    A : ndarray, shape (height, width), optional (default None)
        Image mask

    Returns
    -------
    result_y : ndarray, shape (n_patches, m ** 2)
        Array of flattened patches
    masks : ndarray, shape (n_patches, m ** 2)
        Array of flattened partial masks
    """
    r, c = im.shape
    patches = []
    patches_a = []
    for i in range(r):
        for j in range(c):
            if i + m <= r and j + m <= c:
                patches.append((im[i : i + m, j : j + m]).reshape(m * m, 1))
                if A is not None:
                    patches_a.append(
                        (A[i : i + m, j : j + m]).reshape(m * m, 1)
                    )
    result_y = np.concatenate(patches, axis=1).T
    if A is not None:
        masks = np.concatenate(patches_a, axis=1).T
        return result_y, masks
    else:
        return result_y, (result_y != 0).astype(float)


def patch_average(patches, m, r, c):
    """
    Aggregate overlapping patches to an image by averaging them.

    Parameters
    ----------
    patches : ndarray, shape (n_patches, m ** 2)
        Array of flattened patches
    m : int
        Patch dimension
    r : int
        Image height
    c : int
        Image width

    Returns
    -------
    im : ndarray, shape (height, width)
        Output image
    """
    im = np.zeros((r, c))
    avg = np.zeros((r, c))
    cpt = 0
    for i in range(r):
        for j in range(c):
            if i + m <= r and j + m <= c:
                im[i : i + m, j : j + m] += patches[cpt, :].reshape(m, m)
                avg[i : i + m, j : j + m] += np.ones((m, m))
                cpt += 1
    im = im / avg

    return im


def create_image_digits(width, height, k=0.1, rng=None):
    """
    Generate an image filled with digits.

    Parameters
    ----------
    width : int
        Image width
    height : int
        Image height
    k : float, optional (default 0.1)
        Proportion of void in the image
    rng : numpy.random.Generator, optional (default None)
        Random generator

    Returns
    -------
    image : ndarray, shape (height, width)
        Generated image with values in [0, 1]
    """
    digits = load_digits()
    image = np.zeros((height, width))
    if rng is None:
        rng = np.random.default_rng()
    for i in range(width // 10):
        for j in range(height // 10):
            if rng.random() > k:
                random_img = digits.images[rng.integers(10)]
                j_start, j_end = j * 10 + 1, (j + 1) * 10 - 1
                i_start, i_end = i * 10, i * 10 + 8
                image[j_start:j_end, i_start:i_end] = random_img
    image /= 16

    return image


def rec_score_digits(D, D_ref):
    """Score in [0, 1] which indicates if we recover D_ref in D.

    Parameters
    ----------
    D : ndarray, shape (n_atoms, atom_height, atom_width)
        Convolutional dictionary
    D_ref : ndarray, shape (n_atoms, ref_atom_height, ref_atom_width)
        Reference convolutional dictionary

    Returns
    -------
    score : float
        Returned score
    """
    scores = np.zeros(D_ref.shape[0])
    for i in range(D_ref.shape[0]):
        corr = np.zeros(D.shape[0])
        for j in range(D.shape[0]):
            corr[j] = np.abs(correlate2d(D[j], D_ref[i])).max()
        scores[i] = corr.max()
    score = scores.mean()

    return score


def gaussian_kernel(dim, sigma):
    """Generate a 2D gaussian kernel of given size and standard deviation.

    Parameters
    ----------
    dim : int
        Kernel size
    sigma : float
        Kernel standard deviation

    Returns
    -------
    kernel : ndarray, shape (dim, dim)
        Gaussian kernel of size dim*dim
    """
    t = np.linspace(-1, 1, dim)
    gaussian = np.exp(-0.5 * (t / sigma) ** 2)
    kernel = gaussian[None, :] * gaussian[:, None]
    kernel /= kernel.sum()

    return kernel


def determinist_inpainting(
    img_path, prop, sigma, dtype=torch.float32, device="cpu"
):
    """Apply deterministic inpainting to an image."""
    hash = hashlib.md5(bytes(img_path, "utf-8")).hexdigest()
    seed = int(hash[:8], 16)
    img = Image.open(img_path).convert("L")
    img = T.ToTensor()(img).to(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    mask = (
        torch.rand(img.shape, generator=rng, dtype=dtype, device=device) > prop
    )
    noise = torch.randn(img.shape, generator=rng, dtype=dtype, device=device)

    return img * mask + sigma * noise, mask
