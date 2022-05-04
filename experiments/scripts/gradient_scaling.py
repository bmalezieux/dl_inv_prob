import numpy as np
import torch

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
N_EXP = 10
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
patches = patches.reshape(patches.shape[0], -1)[:N_data].T

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

A = RNG.multivariate_normal(np.zeros(dim_signal), cov, size=(10000, m))
A /= np.sqrt(np.sum(A**2, axis=1, keepdims=True))

true_cov = np.zeros((dim_signal, dim_signal))
for i in range(10000):
    true_cov += A[i].T @ A[i]
true_cov /= 10000

dl = DictionaryLearning(n_components, init_D=D_init,
                        rng=RNG, device=DEVICE)
dl.fit(patches[None, :])
D_true = dl.D_


times_results = []
loss_results = []
scores_results = []

# Unscaled
times_results.append([])
loss_results.append([])
scores_results.append([])

# Scaled
times_results.append([])
loss_results.append([])
scores_results.append([])

# Reg
for _ in range(len(reg_list)):
    times_results.append([])
    loss_results.append([])
    scores_results.append([])


for i in tqdm(range(N_EXP)):

    A = RNG.multivariate_normal(
        np.zeros(dim_signal),
        cov,
        size=(N_matrices, m)
    )
    A /= np.sqrt(np.sum(A**2, axis=1, keepdims=True))
    Y = []
    n_patches = patches.shape[1] // N_matrices
    for i in range(N_matrices):
        Y.append(
            (A[i] @ patches[:, i * n_patches: (i+1) * n_patches])[None, :]
        )
    Y = np.concatenate(Y, axis=0)
    print(Y.shape)

    times, loss, scores = compute_results(Y, A, None, n_components,
                                          D_init, D_true, RNG, DEVICE)
    times_results[0].append(times.copy())
    loss_results[0].append(loss.copy())
    scores_results[0].append(scores.copy())

    times, loss, scores = compute_results(Y, A, np.linalg.inv(true_cov),
                                          n_components, D_init, D_true,
                                          RNG, DEVICE)
    times_results[1].append(times.copy())
    loss_results[1].append(loss.copy())
    scores_results[1].append(scores.copy())

    for j in range(len(reg_list)):
        cov_inv = np.linalg.inv(
            true_cov + reg_list[j] * np.eye(true_cov.shape[0])
        )
        times, loss, scores = compute_results(Y, A, cov_inv, n_components,
                                              D_init, D_true,
                                              RNG, DEVICE)
        times_results[j+2].append(times.copy())
        loss_results[j+2].append(loss.copy())
        scores_results[j+2].append(scores.copy())

new_times = np.linspace(0, 30, 100)
recoveries = []
for i in range(len(times_results)):
    recoveries.append(np.zeros((len(times_results[i]), len(new_times))))

t_max = np.max(new_times)
for i in range(len(times_results)):
    for j in range(len(times_results[i])):
        if times_results[i][j][-1] < t_max:
            times_results[i][j][-1] = t_max
        f = interpolate.interp1d(times_results[i][j], scores_results[i][j])
        recoveries[i][j] = f(new_times)

np.save("../results/scaling_gradient_times.npy", np.array(new_times))
np.save("../results/scaling_gradient_recoveries.npy", np.array(recoveries))
np.save("../results/scaling_gradient_reg_list.npy", np.array(reg_list))
