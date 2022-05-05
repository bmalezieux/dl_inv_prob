import numpy as np
import torch
import pandas as pd

from scipy import interpolate
from scipy.stats import ortho_group
from tqdm import tqdm
from dl_inv_prob.dl import DictionaryLearning
from dl_inv_prob.utils import recovery_score
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d


def compute_results(Y, A, cov_inv, n_components, D_init, D_true, rng, device):

    dl = DictionaryLearning(n_components, init_D=D_init, rng=rng,
                            device=device, keep_dico=True)
    dl.fit(Y, A, cov_inv=cov_inv)

    path_scores = []
    for elt in dl.path_optim:
        path_scores.append(recovery_score(elt, D_true))

    return (np.array(dl.path_times),
            np.array(dl.path_loss),
            np.array(path_scores))


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
N_EXP = 1
RNG = np.random.default_rng(100)
PATH_DATA = "../data/flowers.png"


# Parameters of the signal
dim_patch = 10
dim_signal = dim_patch ** 2
n_components = dim_signal
m = int(0.7 * dim_signal)
N_data = 10000

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

# Parameters experiment
sigma_diag = 0.3
lambd = 0.1
N_matrices = 10
reg_list = np.arange(0.1, 0.5, 0.1)
D_init = patches[:, :n_components]
result_list = []

# Operators
W = ortho_group.rvs(dim_signal)
u = np.maximum(RNG.normal(1, sigma_diag, size=dim_signal), 0.1)
cov = W.T @ np.diag(u) @ W

# A = RNG.multivariate_normal(np.zeros(dim_signal), cov, size=(10000, m))
# A /= np.linalg.norm(A, axis=1, keepdims=True)
# empirical_cov = (A.transpose((0, 2, 1)) @ A).mean(axis=0)
empirical_cov = np.eye(dim_signal)

dl = DictionaryLearning(n_components, init_D=D_init,
                        rng=RNG, device=DEVICE)
dl.fit(patches[None, :])
D_true = dl.D_

results = {
    "times": {"unscaled": [], "scaled": []},
    "loss": {"unscaled": [], "scaled": []},
    "scores": {"unscaled": [], "scaled": []}
}

# Reg
for i in range(len(reg_list)):
    results["times"][f"reg {reg_list[i]}"] = []
    results["loss"][f"reg {reg_list[i]}"] = []
    results["scores"][f"reg {reg_list[i]}"] = []


for i in tqdm(range(N_EXP)):

    # A = RNG.multivariate_normal(
    #     np.zeros(dim_signal),
    #     cov,
    #     size=(N_matrices, m)
    # )
    # A /= np.linalg.norm(A, axis=1, keepdims=True)
    A = np.concatenate([np.eye(dim_signal)[None, :] for i in range(N_matrices)], axis=0)
    Y = []
    n_patches = patches.shape[1] // N_matrices
    for i in range(N_matrices):
        Y.append(
            (A[i] @ patches[:, i * n_patches: (i+1) * n_patches])[None, :]
        )
    Y = np.concatenate(Y, axis=0)

    times, loss, scores = compute_results(Y, A, None, n_components,
                                          D_init, D_true, RNG, DEVICE)
    results["times"]["unscaled"].append(times.copy())
    results["loss"]["unscaled"].append(loss.copy())
    results["scores"]["unscaled"].append(scores.copy())

    times, loss, scores = compute_results(Y, A, np.linalg.inv(empirical_cov),
                                          n_components, D_init, D_true,
                                          RNG, DEVICE)
    results["times"]["scaled"].append(times.copy())
    results["loss"]["scaled"].append(loss.copy())
    results["scores"]["scaled"].append(scores.copy())

    for j in range(len(reg_list)):
        cov_inv = np.linalg.inv(
            empirical_cov + reg_list[j] * np.eye(empirical_cov.shape[0])
        )
        times, loss, scores = compute_results(Y, A, cov_inv, n_components,
                                              D_init, D_true,
                                              RNG, DEVICE)
        results["times"][f"reg {reg_list[j]}"].append(times.copy())
        results["loss"][f"reg {reg_list[j]}"].append(loss.copy())
        results["scores"][f"reg {reg_list[j]}"].append(scores.copy())

results["reg_list"] = {"reg_list": reg_list}
results_df = pd.DataFrame(results)
results_df.to_pickle("../results/gradient_scaling.pickle")
