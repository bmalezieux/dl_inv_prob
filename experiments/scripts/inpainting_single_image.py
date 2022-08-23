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

# from dl_inv_prob.dip import DIPInpainting
from dl_inv_prob.utils import psnr

from joblib import Memory
from pathlib import Path
from tqdm import tqdm

from utils.tv import ProxTV

from utils.wavelets import SparseWavelets

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
    "CDL": ConvolutionalInpainting,
    # "DIP": [DIPInpainting],
    "TV": ProxTV,
    "Wavelets": SparseWavelets,
}

#####################
# Solver generation #
#####################


def generate_algo(params):
    """
    Initiate an algorithm from a solver and its parameters.
    """

    solver_name = params["name"]

    if solver_name == "CDL":
        dim_atom = int(params["size"] * params["prop_atom"])
        solver = SOLVERS[solver_name](
            n_components=params["n_atoms"],
            lambd=params["lambd"],
            atom_height=dim_atom,
            atom_width=dim_atom,
            device=DEVICE,
            rng=NP_RNG,
        )

    elif solver_name == "TV":
        solver = SOLVERS[solver_name](
            lambd=params["lambd"],
            n_iter=params["n_iter"],
        )

    elif solver_name == "Wavelets":
        solver = SOLVERS[solver_name](
            lambd=params["lambd"],
            wavelet=params["wavelet"],
            n_iter=params["n_iter"],
            step=params["step"],
        )

    return solver


#############
# Benchmark #
#############


@mem.cache
def run_test(params):
    """
    Runs several tests using a given set of parameters and a given solver
    """

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

    # Reconstruction
    start = time.time()
    algo = generate_algo(params)
    algo.fit(img_inpainting[None, :, :], mask[None, :, :])
    rec = algo.rec().squeeze()
    rec = np.clip(rec, 0, 1)
    stop = time.time()

    # Result
    psnr_rec = psnr(rec, img)
    psnr_corr = psnr(img_inpainting, img)
    delta = stop - start
    delta = str(datetime.timedelta(seconds=delta))

    solver_name = params["name"]
    n_atoms = params["n_atoms"]
    lambd = params["lambd"]
    prop_atom = params["prop_atom"]
    n_iter = params["n_iter"]
    wavelet = params["wavelet"]
    step = params["step"]

    print(f"rho = {rho}, size = {size}, sigma = {sigma}")
    print(f"solver: {solver_name}")
    if solver_name == "CDL":
        print(f"lambd = {lambd}, n_atoms = {n_atoms}, prop_atom = {prop_atom}")
    elif solver_name == "TV":
        print(f"lambd = {lambd}, n_iter = {n_iter}")
    elif solver_name == "Wavelets":
        print(f"wavelet: {wavelet}")
        print(f"lambd = {lambd}, n_iter = {n_iter}, step = {step}")
    print(f"psnr_rec = {psnr_rec:.2f}, psnr_corr = {psnr_corr:.2f}")
    print(f"time: {delta}\n")

    results = {
        "psnr_corr": psnr_corr,
        "psnr_rec": psnr_rec,
        "time": delta,
    }

    return results


###############
# Main script #
###############


if __name__ == "__main__":

    hyperparams = {
        "name": SOLVERS.keys(),
        "lambd": [0.1],
        "n_atoms": [20],
        "prop_atom": [0.1],
        "rho": [0.9],
        "sigma": [0.1],
        "size": [64],
        "wavelet": ["db3"],
        "n_iter": [1000],
        "step": [1.0],
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

    # Data Frame
    results = pd.DataFrame(dict_results)
    results.to_parquet(os.path.join(RESULTS, "inpainting_single_image.parquet"))
