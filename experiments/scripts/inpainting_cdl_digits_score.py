import numpy as np
import pandas as pd
import torch
import itertools

from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.utils import create_image_digits, rec_score_digits
from sklearn.datasets import load_digits
from joblib import Memory
from tqdm import tqdm

SEED = 2022
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(SEED)

mem = Memory(location="./tmp_cdl_digits/", verbose=0)

# Dictionary containing the ten digits of the image
D_ref = load_digits().images[:10]
D_ref /= np.linalg.norm(D_ref, axis=(1, 2), keepdims=True)

# Fix dimensions
N_ATOMS = 50
DIM_ATOM = 10

# Fix initialization
D_init = RNG.normal(size=(N_ATOMS, 1, DIM_ATOM, DIM_ATOM))


@mem.cache
def run_test(params):

    rho = params["rho"]
    sigma = params["sigma"]
    lambd = params["lambda"]
    size = params["size"]
    n_atoms = N_ATOMS
    dim_atom = DIM_ATOM

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

    result = {
        "score": score,
    }

    return result


if __name__ == "__main__":

    hyperparams = {
        "size": [256, 128],
        "sigma": np.arange(100, 801, 100),
        "rho": np.arange(0.1, 1.1, 0.1),
        "lambda": [0.1, 1.]
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dict_results = {}

    for params in tqdm(permuts_params):
        try:
            results = run_test(params)

            # Storing results

            for key in params.keys():
                if key != "solver":
                    if key not in dict_results:
                        dict_results[key] = [params[key]]
                    else:
                        dict_results[key].append(params[key])

            for key in results:
                if key not in dict_results:
                    dict_results[key] = [results[key]]
                else:
                    dict_results[key].append(results[key])

        except (KeyboardInterrupt, SystemExit):
            raise

    results_df = pd.DataFrame(dict_results)
    results_df.to_csv(
        "experiments/results/inpainting_cdl_digits_score.csv"
    )
