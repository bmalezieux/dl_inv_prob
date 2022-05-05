import numpy as np
import pandas as pd

from scipy.stats import ortho_group
from tqdm import tqdm
from celer import Lasso
from threadpoolctl import threadpool_limits
from joblib import delayed, Parallel
from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image


def compute_corr(Y, true_gradient, m, N_matrices, D_hat, cov, reg_list, lambd, rng):
    with threadpool_limits(limits=1, user_api="blas"):
        # Parameters
        dim_signal = D_hat.shape[0]

        # List of operators
        A = rng.multivariate_normal(
            np.zeros(dim_signal),
            cov,
            size=(N_matrices, m)
        )
        A /= np.linalg.norm(A, axis=1, keepdims=True)

        # Gradients
        gradients = []

        # Unscaled
        gradients.append([])

        # Scaled
        gradients.append([])

        # Reg
        for _ in range(len(reg_list)):
            gradients.append([])

        # Covariances
        covariances_inv = []
        covariances_inv.append(np.linalg.inv(cov))
        for reg in reg_list:
            covariances_inv.append(
                np.linalg.inv(cov + reg * np.eye(cov.shape[0]))
            )
        index = Y.shape[1] // N_matrices
        for k in range(N_matrices):
            y = A[k] @ Y[:, k * index: (k+1) * index]
            lasso = Lasso(alpha=lambd / y.shape[0], fit_intercept=False)
            lasso.fit(A[k] @ D_hat, y)
            z_hat = lasso.coef_.T
            gradients[0].append(
                A[k].T @ (A[k] @ D_hat @ z_hat - y) @ z_hat.T
            )
            gradients[1].append(
                covariances_inv[0] @ A[k].T @ (A[k] @ D_hat @ z_hat - y) @ z_hat.T
            )
            for i, reg in enumerate(reg_list):
                gradients[i+2].append(
                    covariances_inv[i+1] @ A[k].T @ (A[k] @ D_hat @ z_hat - y) @ z_hat.T
                )

        corr = []
        for gradient in gradients:
            g = np.mean(gradient, axis=0)
            g /= np.linalg.norm(g)
            corr.append(np.trace(g.T @ true_gradient))

        return corr


N_EXP = 10
RNG = np.random.default_rng(100)
PATH_DATA = "../data/flowers.png"

# Parameters of the signal
dim_patch = 10
dim_signal = dim_patch ** 2
n_components = dim_signal
m = int(0.7 * dim_signal)

# Image
im = Image.open(PATH_DATA)
im_gray = im.convert("L")
im_gray_resized = im_gray.resize((128, 128), Image.ANTIALIAS)
im_to_process = np.array(im_gray_resized) / 255.

# Patches
patches = extract_patches_2d(im_to_process, (dim_patch, dim_patch))
patches = patches.reshape(patches.shape[0], -1)
RNG.shuffle(patches)
patches = patches.T
N_data = patches.shape[1]

# Parameters experiment
sigma_diag = np.linspace(0, 0.5, 20)
lambd = 0.1
N_matrices = 10
reg_list = np.arange(0.1, 0.5, 0.1)
D_hat = patches[:, :n_components]
result_list = []

# True gradient
lasso = Lasso(alpha=lambd / patches.shape[0], fit_intercept=False)
lasso.fit(D_hat, patches)
z_hat = lasso.coef_.T
true_gradient = (D_hat @ z_hat - patches) @ z_hat.T
true_gradient /= np.linalg.norm(true_gradient)

for sigma in tqdm(sigma_diag):

    W = ortho_group.rvs(dim_signal)
    u = np.maximum(RNG.normal(1, sigma, size=dim_signal), 0.1)
    cov = W.T @ np.diag(u) @ W

    A = RNG.multivariate_normal(np.zeros(dim_signal), cov, size=(10000, m))
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    empirical_cov = (A.transpose((0, 2, 1)) @ A).mean(axis=0)

    results = Parallel(n_jobs=10)(
        delayed(compute_corr)(
            patches.copy(),
            true_gradient.copy(),
            m,
            N_matrices,
            D_hat.copy(),
            empirical_cov.copy(),
            reg_list,
            lambd,
            RNG
        ) for _ in range(N_EXP)
    )
    results = np.array(results)
    result_list.append(results.copy())

results_final = np.array(result_list)

results_df = {
    "results": {"results": results_final},
    "sigma_diag": {"sigma_diag": sigma_diag},
    "reg_list": {"reg_list": reg_list}
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("../results/gradient_scaling_correlation.pickle")
