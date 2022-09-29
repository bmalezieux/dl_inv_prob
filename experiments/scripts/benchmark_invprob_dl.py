import numpy as np
import pandas as pd
import itertools
import os
import torch

from pathlib import Path
from joblib import Memory
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from dl_inv_prob.dl import Inpainting

NUM_EXP = 1
SEED = 2022
EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/invprob_dl.csv")
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
mem = Memory(location='./tmp_invprob_dl/', verbose=0)


def generate_data(dico, N, k=0.3, rng=None):
    """
    Generate data from dictionary

    Parameters
    ----------
    dico : np.array
        dictionary
    N : int
        number of samples
    k : float, optional
        sparsity, by default 0.3
    rng : np.random.Generator

    Returns
    -------
    (np.array, np.array)
        signal, sparse codes
    """
    if rng is None:
        rng = np.random.get_default_rng()
    d = dico.shape[1]
    X = (rng.random((d, N)) > (1-k)).astype(float)
    X *= rng.normal(scale=1, size=(d, N))
    return dico @ X, X


def recovery_score(D, Dref):
    """
    Comparison between a two dictionaries
    """
    cost_matrix = np.abs(Dref.T@D)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    score = cost_matrix[row_ind, col_ind].sum() / D.shape[1]

    return score


def create_mask(N, n, p, rng=None):
    """
    Create random mask for inpainting

    Parameters
    ----------
    N : int
        Number of samples
    n : int
        Dimension signal
    p : float
        ratio of observed data < 1
    rng : np.random.Generator

    Returns
    -------
    np.array(n, N)
        Mask
    """
    if rng is None:
        rng = np.random.get_default_rng()
    omega = rng.random((n, N))
    omega = (omega > (1 - p)).astype(float)
    return omega


@mem.cache
def run_test(params):
    """
    Run NUM_EXP experiments with given parameters

    Parameters
    ----------
    params : dict
        parameters

    Returns
    -------
    dict
        results
    """

    n = params["n"]
    L = params["L"]
    n_samples = params["n_samples"]
    k = params["sparsity"]
    lambd = params["lambda"]
    p = params["p"]
    n_iter = params["n_iter"]
    seed = params["seed"]

    rng = np.random.default_rng(seed)

    D = rng.normal(size=(n, L))
    D /= np.sqrt(np.sum(D**2, axis=0))

    scores = []

    for i in range(NUM_EXP):

        mask = create_mask(n_samples, n, p, rng=rng)
        Y, X = generate_data(D, n_samples, k=k, rng=rng)
        Y_degraded = mask * Y

        D_init = rng.normal(size=(n, L))
        D_init /= np.sqrt(np.sum(D_init**2, axis=0))

        dl = Inpainting(
            n_components=L,
            init_D=D_init,
            max_iter=n_iter,
            lambd=lambd,
            device=DEVICE,
            rng=rng
        )
        dl.fit(Y_degraded[None, :, :], mask[None, :, :])

        D_hat = dl.D_
        scores.append(recovery_score(D, D_hat))

    results = {
        "score_avg": np.mean(scores),
        "score_std": np.std(scores),
        "score_max": np.max(scores),
        "score_min": np.min(scores),
        "NUM_EXP": NUM_EXP
    }

    return results


if __name__ == "__main__":

    hyperparams = {
        "p": np.linspace(0.2, 1, 20),
        "n_samples": np.logspace(2, 4, num=20, dtype=int),
        "sparsity": np.round(
            np.logspace(
                np.log(0.03) / np.log(10),
                np.log(0.3) / np.log(10),
                num=20
            ),
            decimals=2),
        "lambda": [0.1],
        "L": [100],
        "n": [100],
        "n_iter": [100],
        "seed": [SEED]
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dico_results = {}

    for params in tqdm(permuts_params):
        try:
            results = run_test(params)

            # Storing results
            for key in params.keys():
                if key not in dico_results:
                    dico_results[key] = [params[key]]
                else:
                    dico_results[key].append(params[key])

            for key in results.keys():
                if key not in dico_results:
                    dico_results[key] = [results[key]]
                else:
                    dico_results[key].append(results[key])

        except (KeyboardInterrupt, SystemExit):
            raise

    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)
