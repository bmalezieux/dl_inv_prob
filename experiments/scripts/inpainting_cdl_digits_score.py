import datetime
from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.utils import create_image_digits, rec_score_digits
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import time
import torch

SEED = 2022
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(SEED)

n_atoms = 50
dim_atom = 10  # A bit larger than the 8x8 digits

sigmas = [0.0, 0.05, 0.1]
n_sigmas = len(sigmas)
sizes = np.arange(100, 801, 100)
n_sizes = sizes.shape[0]
lambds = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
n_lambds = len(lambds)
rhos = np.arange(0.1, 1.1, 0.1)
n_rhos = len(rhos)
scores = np.zeros((n_sigmas, n_sizes, n_lambds, n_rhos))

print(f"n_sizes = {n_sizes}, n_lambds = {n_lambds}, n_rhos = {n_rhos}\n")

# Dictionary containing the ten digits of the image
D_ref = load_digits().images[:10]
D_ref /= np.linalg.norm(D_ref, axis=(1, 2), keepdims=True)

start_time = time.time()

# Fixed random dictionary initialization
D_init = RNG.normal(size=(n_atoms, 1, dim_atom, dim_atom))

for i_sigma, sigma in enumerate(sigmas):
    for i_lambd, lambd in enumerate(lambds):
        for i_s, rho in enumerate(rhos):
            for i_size, size in enumerate(sizes):
                y = create_image_digits(size, size, k=0.1, rng=RNG)
                mask = RNG.binomial(1, rho, y.shape)
                # Corrupt the image
                y_inpainting = y * mask + sigma * RNG.normal(size=y.shape)

                dl = ConvolutionalInpainting(
                    n_atoms,
                    lambd=lambd,
                    init_D=D_init,
                    atom_height=dim_atom,
                    atom_width=dim_atom,
                    device=DEVICE,
                    rng=RNG,
                    alpha=0.1,
                )
                # Perform inpainting with CDL
                dl.fit(y_inpainting[None, :, :], mask[None, :, :])
                D = dl.D_.squeeze()
                score = rec_score_digits(D, D_ref)

                scores[i_sigma, i_size, i_lambd, i_s] = score
                print(f"size = {size}, rho = {rho}, sigma = {sigma}")
                print(f"lambd = {lambd:.2f}")
                print(f"score = {score:.2f}")
                delta = time.time() - start_time
                delta = str(datetime.timedelta(seconds=delta))
                print(f"elapsed time: {delta}\n")

results_df = {
    "scores": {"scores": scores},
    "sizes": {"sizes": sizes},
    "lambds": {"lambds": lambds},
    "rhos": {"rhos": rhos},
    "sigmas": {"sigmas": sigmas},
}
results_df = pd.DataFrame(results_df)
results_df.to_parquet(
    "experiments/results/inpainting_cdl_digits_score.parquet"
)
