import numpy as np
import torch
import os
import itertools
import pandas as pd

from dl_inv_prob.dl import DictionaryLearning
from dl_inv_prob.utils import generate_dico, generate_data, recovery_score
from pathlib import Path
from joblib import Memory
from tqdm import tqdm

N_EXP = 5
SEED = 2022
EXPERIMENTS = Path(__file__).resolve().parents[1]
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(EXPERIMENTS, "results/partial_rec.csv")
mem = Memory(location="./tmp_partial_rec/", verbose=0)


@mem.cache
def run_test(params):

    scores = []
    seed = params["seed"]
    rng = np.random.default_rng(seed)
    sparsity = params["sparsity"]
    dim_measurement = params["dim_measurement"]
    n_components = params["n_components"]
    dim_signal = params["dim_signal"]
    n_data = params["n_data"]

    for _ in range(N_EXP):

        D = generate_dico(n_components, dim_signal, rng=rng)
        D_init = generate_dico(n_components, dim_signal, rng=rng)

        A = rng.normal(size=(1, dim_measurement, dim_signal))
        A /= np.linalg.norm(A, axis=1, keepdims=True)

        Y, _ = generate_data(A @ D, N=n_data, s=sparsity, rng=rng)

        dl = DictionaryLearning(
            n_components,
            init_D=D_init,
            device=DEVICE,
            rng=rng,
            lambd=params["lambd"],
        )
        dl.fit(Y, A)

        D_sol = dl.D_
        scores.append(recovery_score(D_sol, D))

    results = {
        "score_avg": np.mean(scores),
        "score_q1": np.quantile(scores, q=0.25),
        "score_q3": np.quantile(scores, q=0.75),
    }

    return results


if __name__ == "__main__":

    hyperparams = {
        "sparsity": [0.03, 0.1, 0.2, 0.3],
        "dim_measurement": np.linspace(10, 100, 20, dtype=int),
        "lambd": [0.01, 0.05, 0.1],
        "seed": [SEED],
        "n_components": [100],
        "dim_signal": [100],
        "n_data": [10000],
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
