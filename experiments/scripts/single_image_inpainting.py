"""
Benchmark for inpainting with convolutional dictionary learning.
"""
import itertools
from dl_inv_prob.utils import determinist_inpainting
import numpy as np
import os
import pandas as pd
import random
import torch

from dl_inv_prob.common_utils import (
    torch_to_np,
)
from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.dip import DIPInpainting
from dl_inv_prob.utils import psnr, is_divergence
from joblib import Memory
from pathlib import Path
from tqdm import tqdm
from utils.tv import ProxTV
from utils.wavelets import SparseWavelets

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = os.path.join(EXPERIMENTS, "data")
# IMG = os.path.join(DATA, "flowers.png")
RESULTS = os.path.join(EXPERIMENTS, "results")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULT_FILE = "inpainting_single_image.csv"

IMGS = {}
for file in os.listdir(DATA):
    if file.endswith("png") or file.endswith("jpg"):
        name = file.split(".")[0]
        img_path = os.path.join(DATA, file)
        IMGS.update([(name, img_path)])

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
    "TV": ProxTV,
    "Wavelets": SparseWavelets,
    "DIP": DIPInpainting,
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
        "rho": params["rho"],
        "sigma": params["sigma"],
    }

    solver_name = params["name"]

    if solver_name == "CDL":
        params_solver["lambd"] = params["lambd"]
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


@mem.cache
def run_test(params):
    """
    Runs several tests using a given set of parameters and a given solver
    """

    # Data generation
    size = params["size"]
    rho = params["rho"]
    sigma = params["sigma"]
    IMG = IMGS[params["image"]]
    img, img_inpainting, mask = determinist_inpainting(
        IMG, prop=1 - rho, sigma=sigma, size=size
    )
    img = torch_to_np(img).squeeze()
    img_inpainting = torch_to_np(img_inpainting).squeeze()
    mask = torch_to_np(mask).squeeze()

    # Reconstruction
    algo = generate_algo(params)
    algo.fit(img_inpainting[None, :, :], mask[None, :, :])
    rec = algo.rec().squeeze()
    rec = np.clip(rec, 0, 1)

    # Result
    psnr_rec = psnr(rec, img)
    psnr_rec_ker = psnr(rec[mask == 0], img[mask == 0])
    psnr_rec_range = psnr(rec[mask == 1], img[mask == 1])
    is_rec = is_divergence(rec, img)

    results = {
        "psnr_rec": psnr_rec,
        "psnr_rec_ker": psnr_rec_ker,
        "psnr_rec_range": psnr_rec_range,
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
        "sigma": [0.0, 0.02, 0.05, 0.1],
        "rho": np.arange(0.1, 1.0, 0.1),
        "name": SOLVERS.keys(),
        "n_atoms": [50],
        "lambd": [0.01, 0.05, 0.1],
        "dim_atoms": [10],
        "n_iter": [1000],
        "wavelet": ["db3"],
        "step": [1.0],
        "lr": [0.01],
        "sigma_input_noise": [1.0],
        "sigma_reg_noise": [0.03],
        "input_depth": [32],
        "model": ["SkipNet"],
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
