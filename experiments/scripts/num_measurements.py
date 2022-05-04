import numpy as np
import torch

from tqdm import tqdm
from dl_inv_prob.dl import DictionaryLearning
from dl_inv_prob.utils import generate_dico, generate_data, recovery_score

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
N_EXP = 10
RNG = np.random.default_rng(100)

m_values = [40, 60, 80, 100]
N_mat_values = np.arange(1, 6, 1)

dim_signal = 100
n_components = 100
N_data_total = 10000

scores = np.zeros((N_EXP, len(m_values), len(N_mat_values)))


for n_exp in tqdm(range(N_EXP)):

    D = generate_dico(n_components, dim_signal, rng=RNG)
    D_init = generate_dico(n_components, dim_signal, rng=RNG)

    for i, m in enumerate(m_values):
        for j, N_matrices in enumerate(N_mat_values):

            A = RNG.normal(size=(N_matrices, m, dim_signal))
            A /= np.linalg.norm(A, axis=1, keepdims=True)

            Y, _ = generate_data(A @ D, N_data_total // N_matrices, rng=RNG)

            dl = DictionaryLearning(
                n_components,
                init_D=D_init,
                device=DEVICE,
                rng=RNG
            )
            dl.fit(Y, A)

            D_hat = dl.D_
            scores[n_exp, i, j] = recovery_score(D, D_hat)


np.save("../results/compressed_sensing_n_matrices.npy", scores)
np.save("../results/compressed_sensing_m_values.npy", m_values)
np.save("../results/compressed_sensing_N_mat_values.npy", N_mat_values)
