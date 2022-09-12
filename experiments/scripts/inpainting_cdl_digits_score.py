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

mem = Memory(location="./tmp_cdl_digits/", verbose=0)

# Dictionary containing the ten digits of the image
D_ref = load_digits().images[:10]
D_ref /= np.linalg.norm(D_ref, axis=(1, 2), keepdims=True)

NUM_EXP = 5

@mem.cache
def run_test(params, random_seed, num_exp):

    rho = params["rho"]
    sigma = params["sigma"]
    lambd = params["lambda"]
    size = params["size"]
    n_atoms = params["n_atoms"]
    dim_atom = params["dim_atom"]
    rng = np.random.default_rng(random_seed)
    scores = []

    for i in range(num_exp):

        y = create_image_digits(size, size, k=0.1, rng=rng)
        mask = rng.binomial(1, rho, y.shape)
        D_init = rng.normal(size=(n_atoms, 1, dim_atom, dim_atom))
        # Corrupt the image
        y_inpainting = y * mask + sigma * rng.normal(size=y.shape)

        dl = ConvolutionalInpainting(
            n_atoms,
            lambd=lambd,
            init_D=D_init,
            atom_height=dim_atom,
            atom_width=dim_atom,
            device=DEVICE,
            rng=rng,
            alpha=0.1,
        )
        # Perform inpainting with CDL
        dl.fit(y_inpainting[None, :, :], mask[None, :, :])
        D = dl.D_.squeeze()
        scores.append(rec_score_digits(D, D_ref))

    result = {
        "score": np.mean(scores),
        "std": np.std(scores)
    }

    return result


if __name__ == "__main__":

    hyperparams = {
        "size": np.arange(50, 300, 25),
        "sigma": [0., 0.1],
        "rho": [0.3, 0.5, 0.7, 0.9],
        "lambda": [0.01, 0.1, 1.],
        "n_atoms": [30],
        "dim_atom": [10]
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dict_results = {}

    for params in tqdm(permuts_params):
        try:
            results = run_test(params, SEED, NUM_EXP)

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
        "../results/inpainting_cdl_digits_score.csv"
    )
