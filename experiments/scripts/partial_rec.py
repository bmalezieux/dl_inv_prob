import numpy as np
import torch

from tqdm import tqdm
from dl_inv_prob.dl import DictionaryLearning
from dl_inv_prob.utils import generate_dico, generate_data, recovery_score


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
N_EXP = 10
RNG = np.random.default_rng(100)

dim_signal = 100
n_components = 100
N_data_total = 10000

D = generate_dico(n_components, dim_signal, rng=RNG)
D_init = generate_dico(n_components, dim_signal, rng=RNG)

spars = [0.03, 0.1, 0.2, 0.3]
dim_m = np.linspace(10, n_components, 20, dtype=int)

scores = np.zeros((N_EXP, len(spars), len(dim_m)))

for k in tqdm(range(N_EXP)):
    for i, s in enumerate(spars):
        for j, m in enumerate(dim_m):

            A = RNG.normal(size=(1, m, dim_signal))
            A /= np.linalg.norm(A, axis=1, keepdims=True)

            Y, _ = generate_data(A @ D, N=N_data_total, s=s, rng=RNG)

            dl = DictionaryLearning(
                n_components,
                init_D=D_init,
                device=DEVICE,
                rng=RNG
            )
            dl.fit(Y, A)

            D_sol = dl.D_
            scores[k, i, j] = recovery_score(D_sol, D)

np.save("../results/scores_partial.npy", scores)
np.save("../results/dim_m_partial.npy", dim_m)
np.save("../results/spars_partial.npy", spars)
