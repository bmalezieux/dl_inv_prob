"""
Benchmark for inpainting and deblurring on a single image.
"""
import itertools
from dl_inv_prob.utils import determinist_blurr
import numpy as np
import os
import pandas as pd
import random
import torch

from dl_inv_prob.common_utils import (
    torch_to_np,
)
from dl_inv_prob.dl import Deconvolution, TVDeconvolution
from dl_inv_prob.dip import DIPDeblurring
from dl_inv_prob.utils import psnr, is_divergence, split_psnr
from joblib import Memory
from pathlib import Path
from tqdm import tqdm
from utils.tv import ProxTVDeblurring
from utils.wavelets import SparseWaveletsDeblurring

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = os.path.join(EXPERIMENTS, "data")
# IMG = os.path.join(DATA, "flowers.png")
RESULTS = os.path.join(EXPERIMENTS, "results")
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
RESULT_FILE = "deblurring_tv_atoms.csv"

IMGS = {}
# for file in os.listdir(DATA):
#     if file.endswith("png") or file.endswith("jpg"):
#         name = file.split(".")[0]
#         img_path = os.path.join(DATA, file)
#         IMGS.update([(name, img_path)])
IMGS.update([("flowers", os.path.join(DATA, "flowers.png"))])

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


# mem = Memory(location="./tmp_tv_atoms/", verbose=0)


###########
# Solvers #
###########


SOLVERS = {
    "CDL": Deconvolution,
    # "TV": ProxTVDeblurring,
    # "Wavelets": SparseWaveletsDeblurring,
    # "DIP": DIPDeblurring,
    "CDL_TV": TVDeconvolution,
}


#########################
# Parameters extraction #
#########################


def extract_params(params):
    """
    Extract relevant parameters.
    """

    params_solver = {
        "image": params["image"],
        "name": params["name"],
        "size": params["size"],
        "sigma": params["sigma"],
        "size_blurr": params["size_blurr"],
        "sigma_blurr": params["sigma_blurr"],
    }

    solver_name = params["name"]

    if solver_name == "CDL":
        params_solver["lambd"] = params["lambd"]
        params_solver["dim_atoms"] = params["dim_atoms"]
        params_solver["n_atoms"] = params["n_atoms"]

    elif solver_name == "CDL_TV":
        params_solver["lambd"] = params["lambd"]
        params_solver["mu"] = params["mu"]
        params_solver["dim_atoms"] = params["dim_atoms"]
        params_solver["n_atoms"] = params["n_atoms"]

    elif solver_name == "TV":
        params_solver["lambd"] = params["lambd"]
        params_solver["n_iter"] = params["n_iter"]

    elif solver_name == "Wavelets":
        params_solver["lambd"] = params["lambd"]
        params_solver["n_iter"] = params["n_iter"]
        params_solver["step"] = params["step"]
        params_solver["wavelet"] = params["wavelet"]

    elif solver_name == "DIP":
        params_solver["model"] = params["model"]
        params_solver["n_iter"] = params["n_iter"]
        params_solver["lr"] = params["lr"]
        params_solver["sigma_input_noise"] = params["sigma_input_noise"]
        params_solver["sigma_reg_noise"] = params["sigma_reg_noise"]
        params_solver["input_depth"] = params["input_depth"]

    return params_solver


#####################
# Solver generation #
#####################


def generate_algo(params):
    """
    Initiate an algorithm from parameters.
    """

    solver_name = params["name"]

    if solver_name == "CDL":
        dim_atoms = params["dim_atoms"]
        solver = SOLVERS[solver_name](
            n_components=params["n_atoms"],
            lambd=params["lambd"],
            atom_height=dim_atoms,
            atom_width=dim_atoms,
            device=DEVICE,
            rng=NP_RNG,
        )

    if solver_name == "CDL_TV":
        dim_atoms = params["dim_atoms"]
        solver = SOLVERS[solver_name](
            n_components=params["n_atoms"],
            lambd=params["lambd"],
            mu=params["mu"],
            atom_height=dim_atoms,
            atom_width=dim_atoms,
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

    elif solver_name == "DIP":
        solver = SOLVERS[solver_name](
            model=params["model"],
            n_iter=params["n_iter"],
            lr=params["lr"],
            sigma_input_noise=params["sigma_input_noise"],
            sigma_reg_noise=params["sigma_reg_noise"],
            input_depth=params["input_depth"],
            device=DEVICE,
        )

    return solver


#############
# Benchmark #
#############


def data_generation(params):

    size = params["size"]
    sigma_blurr = params["sigma_blurr"]
    size_blurr = params["size_blurr"]
    sigma = params["sigma"]
    IMG = IMGS[params["image"]]
    img, img_blurred, blurr = determinist_blurr(
        IMG, sigma_blurr, size_blurr, sigma=sigma, size=size
    )
    img = torch_to_np(img).squeeze()
    img_corrupted = torch_to_np(img_blurred).squeeze()
    A = blurr.numpy()

    return img, img_corrupted, A


def run_test(params):
    """
    Runs several tests using a given set of parameters and a given solver
    """

    # Data generation
    img, img_corrupted, A = data_generation(params)

    # Reconstruction
    algo = generate_algo(params)
    algo.fit(img_corrupted[None, :, :], A[None, :, :])
    rec = algo.rec().squeeze()
    rec = np.clip(rec, 0, 1)

    # Result
    psnr_rec = psnr(rec, img)
    is_rec = is_divergence(rec, img)
    psnr_ran, psnr_ker = split_psnr(rec, img, A[0], params["sigma_blurr"])

    results = {
        "psnr_rec": psnr_rec,
        "psnr_ran": psnr_ran,
        "psnr_ker": psnr_ker,
        "is_rec": is_rec,
    }

    return results


###############
# Main script #
###############


if __name__ == "__main__":

    hyperparams = {
        "image": IMGS.keys(),
        "size": [256],
        "sigma": [0.0],  # [0.0, 0.02, 0.05, 0.1],
        "sigma_blurr": [0.1],  # np.arange(0.1, 1.0, 0.1),
        "size_blurr": [10],
        "n_atoms": [50],
        "lambd": [0.01, 0.05, 0.1],
        "mu": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0],
        "dim_atoms": [10, 20],
        "n_iter": [1000],
        "wavelet": ["db3"],
        "step": [1.0],
        "lr": [0.01],
        "sigma_input_noise": [1.0],
        "sigma_reg_noise": [0.03],
        "input_depth": [32],
        "model": ["SkipNet"],
        "name": SOLVERS.keys(),
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dict_results = {}

    for params in tqdm(permuts_params):
        try:
            params_solver = extract_params(params)
            results = run_test(params_solver)

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
    results.to_csv(os.path.join(RESULTS, RESULT_FILE))
