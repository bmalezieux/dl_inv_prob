"""
Benchmark for inpainting with convolutional dictionary learning.
"""
import datetime
import itertools
from dl_inv_prob.utils import determinist_inpainting
import numpy as np
import os
import pandas as pd
import random
import time
import torch

from dl_inv_prob.common_utils import (
    torch_to_np,
)
from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.utils import psnr

from joblib import Memory
from pathlib import Path
from tqdm import tqdm

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = os.path.join(EXPERIMENTS, "data")
IMG = os.path.join(DATA, "flowers.png")
RESULTS = os.path.join(EXPERIMENTS, "results")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Reproducibility
SEED = 2022
NP_RNG = np.random.default_rng(SEED)
RNG = torch.Generator(device=DEVICE)
RNG.manual_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.use_deterministic_algorithms(True)
np.random.seed(SEED)
random.seed(SEED)

mem = Memory(location="./tmp_inpainting/", verbose=0)

###########
# Solvers #
###########


SOLVERS = {
    "CDL": [ConvolutionalInpainting],
}

#########################
# Parameters generation #
#########################


def generate_params_solver(lambd, name, n_atoms, prop_atom):

    params_solver = {
        "lambd": lambd,
        "name": name,
        "solver": SOLVERS[name][0],
        "n_atoms": n_atoms,
        "prop_atom": prop_atom,
    }

    return params_solver


#############
# Benchmark #
#############


@mem.cache
def run_test(params):
    """
    Runs several tests using a given set of parameters and a given solver
    """

    params_solver = generate_params_solver(
        params["lambd"],
        params["name"],
        params["n_atoms"],
        params["prop_atom"],
    )
    solver = params_solver["solver"]

    # Data generation
    size = params["size"]
    rho = params["rho"]
    sigma = params["sigma"]
    img, img_inpainting, mask = determinist_inpainting(
        IMG, prop=1 - rho, sigma=sigma, size=size
    )
    img = torch_to_np(img).squeeze()
    img_inpainting = torch_to_np(img_inpainting).squeeze()
    mask = torch_to_np(mask).squeeze()

    n_atoms = params_solver["n_atoms"]
    lambd = params_solver["lambd"]
    prop_atom = params_solver["prop_atom"]
    dim_atom = int(size * prop_atom)

    start = time.time()
    algo = solver(
        n_components=n_atoms,
        lambd=lambd,
        atom_height=dim_atom,
        atom_width=dim_atom,
        device=DEVICE,
        rng=NP_RNG,
    )
    algo.fit(img_inpainting[None, :, :], mask[None, :, :])
    rec = algo.rec(img_inpainting[None, :, :]).squeeze()
    rec = np.clip(rec, 0, 1)
    stop = time.time()

    psnr_rec = psnr(rec, img)
    psnr_corr = psnr(img_inpainting, img)
    delta = stop - start
    delta = str(datetime.timedelta(seconds=delta))

    print(f"rho = {rho}, size = {size}, sigma = {sigma}")
    print(f"lambd = {lambd}, n_atoms = {n_atoms}, prop_atom = {prop_atom}")
    print(f"psnr_rec = {psnr_rec:.2f}, psnr_corr = {psnr_corr:.2f}")
    print(f"time: {delta}\n")

    results = {
        "psnr_corr": psnr_corr,
        "psnr_rec": psnr_rec,
        "time": delta,
    }

    return results, params_solver


###############
# Main script #
###############


if __name__ == "__main__":

    hyperparams = {
        "name": SOLVERS.keys(),
        "lambd": [0.1, 0.2, 0.3],
        "n_atoms": [20],
        "prop_atom": [0.1],
        "rho": np.arange(0.1, 1.1, 0.1),
        "sigma": [0.0, 0.05, 0.1, 0.15, 0.2],
        "size": [64, 128, 256],
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dict_results = {}

    for params in tqdm(permuts_params):
        try:
            results, params_solver = run_test(params)

            # Storing results

            for key in params_solver.keys():
                if key != "solver":
                    if key not in dict_results:
                        dict_results[key] = [params_solver[key]]
                    else:
                        dict_results[key].append(params_solver[key])

            for key in results:
                if key not in dict_results:
                    dict_results[key] = [results[key]]
                else:
                    dict_results[key].append(results[key])

            for key in ["rho", "sigma", "size"]:
                if key not in dict_results:
                    dict_results[key] = [params[key]]
                else:
                    dict_results[key].append(params[key])

        except (KeyboardInterrupt, SystemExit):
            raise

    # Data Frame
    results = pd.DataFrame(dict_results)
    results.to_parquet(os.path.join(RESULTS, "inpainting_cdl.parquet"))
