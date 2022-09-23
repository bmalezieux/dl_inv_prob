import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate2d, convolve
from sklearn.datasets import load_digits
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import hashlib
import itertools


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


def recovery_score(D, D_ref, weights=None):
    """
    Compute a similarity score in [0, 1] between two dictionaries.

    Parameters
    ----------
    D : ndarray, shape (dim_signal, n_components)
        Dictionary
    Dref : ndarray, shape (dim_signal, n_components)
        Reference dictionary
    weights : ndarray, shape (n_components)
        Weights of usage

    Returns
    -------
    score : float
        _Recovery score in [0, 1]
    """
    if weights is None:
        weights = np.ones(D.shape[1])

    cost_matrix = np.abs(D_ref.T @ ((D.T * weights[:, None]).T))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    score = cost_matrix[row_ind, col_ind].sum() / weights.sum()

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
            res[i_patch, :, :] = img[i:i + dim_patch, j:j + dim_patch]
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


def psnr(rec, ref, d=1.):
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
    psnr = 10 * np.log10(d ** 2 / mse)

    return psnr


def split_psnr(rec, img, kernel, sigma):
    size_x, size_y = img.shape

    dirac_image = np.zeros((size_x, size_y))
    dirac_image[size_x // 2, size_y // 2] = 1

    result = convolve(dirac_image, kernel, mode="same")

    f_dirac = np.fft.fft2(result)
    mag_dirac = np.abs(f_dirac)

    value = np.exp(-0.5 * 4) / np.sqrt(2 * np.pi * sigma ** 2)

    ran_kernel = (mag_dirac > value)
    ker_kernel = (mag_dirac < value)

    f_rec = np.fft.fft2(rec)
    f_img = np.fft.fft2(img)

    rec_ran = np.clip(np.abs(np.fft.ifft2(f_rec * ran_kernel)), 0, 1)
    rec_ker = np.clip(np.abs(np.fft.ifft2(f_rec * ker_kernel)), 0, 1)

    img_ran = np.clip(np.abs(np.fft.ifft2(f_img * ran_kernel)), 0, 1)
    img_ker = np.clip(np.abs(np.fft.ifft2(f_img * ker_kernel)), 0, 1)

    return psnr(rec_ran, img_ran, d=0.95), psnr(rec_ker, img_ker, d=0.05)


def is_divergence(rec, ref):
    """
    Compute the Itakura Saito divergence between the spectrum of two images

    Parameters
    ----------
    rec : ndarray, shape (height, width)
        reconstructed image
    ref : ndarray, shape (height, width)
        original image

    Returns
    -------
    is_div : float
        IS divergence
    """

    f_rec = np.fft.fft2(rec)
    mag_rec = np.abs(f_rec)

    f_ref = np.fft.fft2(ref)
    mag_ref = np.abs(f_ref)

    ratio = (mag_rec / mag_ref - np.log(mag_rec / mag_ref) - 1)
    is_div = ratio[~np.isnan(ratio)].mean()

    return is_div


def discrepancy_measure(y):
    """
    Discrepancy measure in a spectrum

    Parameters
    ----------
    y : ndarray, shape (height, width)
        image

    Returns
    -------
    float
        score
    """

    u = np.log(np.abs(np.fft.fft2(y)))
    u /= np.linalg.norm(u)
    v = np.zeros(u.shape[0] + u.shape[1])
    s = 0
    for i, j in itertools.product(
        np.arange(u.shape[0]),
        np.arange(u.shape[1]),
    ):
        v[i + j] += u[i, j]
    for i, j in itertools.product(
        np.arange(v.shape[0]),
        np.arange(v.shape[0]),
    ):
        s += v[i] * v[j] * (
            np.abs(np.log(1 + i) - np.log(1 + j))
        )
    return s


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
                patches.append((im[i:i + m, j:j + m]).reshape(m * m, 1))
                if A is not None:
                    patches_a.append(
                        (A[i:i + m, j:j + m]).reshape(m * m, 1)
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
                im[i:i + m, j:j + m] += patches[cpt, :].reshape(m, m)
                avg[i:i + m, j:j + m] += np.ones((m, m))
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
    img_path, prop, sigma, seed=2022, size=None,
    dtype=torch.float32, device="cpu"
):
    """Apply deterministic inpainting to an image."""
    img = Image.open(img_path).convert("L")
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    img = T.ToTensor()(img).to(device)

    # Generate same mask for each image from seed
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    mask = (
        torch.rand(img.shape, generator=rng, dtype=dtype, device=device) > prop
    )

    # Generate noise for a specific image
    hash = hashlib.md5(bytes(img_path, "utf-8")).hexdigest()
    seed_noise = int(hash[:8], 16)
    rng_noise = torch.Generator(device=device)
    rng_noise.manual_seed(seed_noise)
    noise = torch.randn(img.shape, generator=rng_noise,
                        dtype=dtype, device=device)

    # Degraded image
    img_corr = torch.clip(mask * (img + sigma * noise), 0, 1)

    return img, img_corr, mask


def determinist_blurr(
    img_path, sigma_blurr, size_kernel, sigma, size=None,
    dtype=torch.float32, device="cpu", padding=None
):
    """Apply deterministic blurr to an image."""
    img = Image.open(img_path).convert("L")
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    img = T.ToTensor()(img).to(device)

    # Generate same blurr for each image from seed
    t = np.linspace(-1, 1, size_kernel)
    if sigma_blurr == 0:
        sigma_blurr = 1e-2
    gaussian = np.exp(-0.5 * (t / sigma_blurr) ** 2)
    kernel = gaussian[None, :] * gaussian[:, None]
    kernel /= kernel.sum()
    kernel = torch.tensor(
        kernel,
        device=device,
        dtype=torch.float,
        requires_grad=False
    )

    # Generate noise for a specific image
    hash = hashlib.md5(bytes(img_path, "utf-8")).hexdigest()
    seed_noise = int(hash[:8], 16)
    rng_noise = torch.Generator(device=device)
    rng_noise.manual_seed(seed_noise)

    # Degraded image
    if padding is None:
        img_blurred = F.conv_transpose2d(
            img[None, :, :],
            kernel[None, None, :, :],
        )
    elif padding == "same":
        img_blurred = F.conv2d(
            img[None, :, :],
            torch.flip(kernel[None, None, :, :], dims=[2, 3]),
            padding="same"
        )
    noise = torch.randn(img_blurred.shape, generator=rng_noise,
                        dtype=dtype, device=device)
    img_corr = torch.clip(img_blurred + sigma * noise, 0, 1)

    return img, img_corr[0], kernel[None, :, :]
