"""
Benchmark for inpainting with convolutional dictionary learning.
"""
import datetime
import itertools
from dl_inv_prob.utils import determinist_inpainting, determinist_blurr
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import argparse

from dl_inv_prob.common_utils import (
    torch_to_np,
)
from dl_inv_prob.dl import ConvolutionalInpainting, Deconvolution
from dl_inv_prob.utils import psnr, is_divergence
from joblib import Memory
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('benchmark')
args = parser.parse_args()

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = os.path.join(EXPERIMENTS, "data")
IMG = os.path.join(DATA, "flowers.png")
RESULTS = os.path.join(EXPERIMENTS, "results")
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BENCHMARK_MODE = args.benchmark

if BENCHMARK_MODE == "inpainting":
    RESULT_FILE = "inpainting_single_image.csv"
elif BENCHMARK_MODE == "deblurring":
    RESULT_FILE = "deblurring_single_image.csv"
else:
    print("Not a benchmark")
    exit(0)

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

if BENCHMARK_MODE == "inpainting":
    mem = Memory(location="./tmp_inpainting_supervised/", verbose=0)
elif BENCHMARK_MODE == "deblurring":
    mem = Memory(location="./tmp_deblurring_supervised/", verbose=0)


###########
# Solvers #
###########


if BENCHMARK_MODE == "inpainting":
    SOLVERS = {
        "CDL_supervised": ConvolutionalInpainting,
    }
elif BENCHMARK_MODE == "deblurring":
    SOLVERS = {
        "CDL_supervised": Deconvolution,
    }


#########################
# Parameters extraction #
#########################


def extract_params(params):
    """
    Extract relevant parameters for reconstruction
    """

    if BENCHMARK_MODE == "inpainting":
        params_solver = {
            "name": params["name"],
            "size": params["size"],
            "rho": params["rho"],
            "sigma": params["sigma"],
            "size_blurr": 0,
            "sigma_blurr": 0,
        }
    elif BENCHMARK_MODE == "deblurring":
        params_solver = {
            "name": params["name"],
            "size": params["size"],
            "rho": 0,
            "sigma": params["sigma"],
            "size_blurr": params["size_blurr"],
            "sigma_blurr": params["sigma_blurr"],
        }

    solver_name = params["name"]

    if solver_name == "CDL_supervised":
        params_solver["lambd"] = params["lambd"]
        params_solver["dim_atoms"] = params["dim_atoms"]
        params_solver["n_atoms"] = params["n_atoms"]

    return params_solver


def extract_params_learning(params):
    """
    Extract relevant parameters for supervised learning
    """
    params_solver = {
        "name": params["name"],
        "size": params["size"],
        "size_blurr": params["size_blurr"]
    }

    if params["name"] == "CDL_supervised":
        params_solver["lambd"] = params["lambd"]
        params_solver["dim_atoms"] = params["dim_atoms"]
        params_solver["n_atoms"] = params["n_atoms"]

    return params_solver


#####################
# Solver generation #
#####################


def generate_algo(params):
    """
    Initiate an algorithm from parameters.
    """

    solver_name = params["name"]

    if solver_name == "CDL_supervised":
        dim_atoms = params["dim_atoms"]
        solver = SOLVERS[solver_name](
            n_components=params["n_atoms"],
            lambd=params["lambd"],
            atom_height=dim_atoms,
            atom_width=dim_atoms,
            device=DEVICE,
            rng=NP_RNG,
        )

    return solver


#############
# Benchmark #
#############


def data_generation(params):

    if BENCHMARK_MODE == "inpainting":
        size = params["size"]
        rho = params["rho"]
        sigma = params["sigma"]
        img, img_inpainting, mask = determinist_inpainting(
            IMG, prop=1 - rho, sigma=sigma, size=size
        )
        img = torch_to_np(img).squeeze()
        img_corrupted = torch_to_np(img_inpainting).squeeze()
        A = torch_to_np(mask).squeeze()

    elif BENCHMARK_MODE == "deblurring":
        size = params["size"]
        sigma_blurr = params["sigma_blurr"]
        size_blurr = params["size_blurr"]
        sigma = params["sigma"]
        img, img_blurred, blurr = determinist_blurr(
            IMG, sigma_blurr, size_blurr, sigma=sigma, size=size
        )
        img = torch_to_np(img).squeeze()
        img_corrupted = torch_to_np(img_blurred).squeeze()
        A = blurr.numpy()

    return img, img_corrupted, A


@mem.cache
def learn_dictionary(params_dictionary):
    """
    Learn a dictionary for a given set of parameters
    """

    if BENCHMARK_MODE == "inpainting":
        size = params_dictionary["size"]
        img, img_inpainting, mask = determinist_inpainting(
            IMG, prop=1, sigma=0, size=size
        )
        img = torch_to_np(img).squeeze()
        mask = torch_to_np(mask).squeeze()
        algo = generate_algo(params_dictionary)
        algo.fit(img[None, :, :], np.ones_like(img[None, :, :]))

    elif BENCHMARK_MODE == "deblurring":
        size = params_dictionary["size"]
        size_blurr = params_dictionary["size_blurr"]
        img, img_corrupted, blurr = determinist_blurr(
            IMG, 0, size_blurr, sigma=0, size=size
        )
        img = torch_to_np(img).squeeze()
        algo = generate_algo(params_dictionary)
        algo.fit(img[None, :, :], np.array(blurr)[None, :, :])

    return algo


@mem.cache
def run_test(params):
    """
    Runs several tests using a given set of parameters and a given solver
    """

    # Data generation
    img, img_corrupted, A = data_generation(params)

    params_learning = extract_params_learning(params)
    algo = learn_dictionary(params_learning)

    # Reconstruction
    start = time.time()
    rec = algo.rec(img_corrupted[None, :, :], A[None, :, :]).squeeze()
    rec = np.clip(rec, 0, 1)
    stop = time.time()

    # Result
    psnr_rec = psnr(rec, img)
    # psnr_corr = psnr(img_corrupted, img)
    is_rec = is_divergence(rec, img)
    # is_corr = is_divergence(img_corrupted, img)
    delta = stop - start
    delta = str(datetime.timedelta(seconds=delta))

    results = {
        # "psnr_corr": psnr_corr,
        "psnr_rec": psnr_rec,
        "is_rec": is_rec,
        # "is_corr": is_corr,
        "time": delta,
    }

    return results


###############
# Main script #
###############


if __name__ == "__main__":

    hyperparams = {
        "size": [256],
        "sigma": [0.0, 0.02, 0.05, 0.1],
        "rho": np.arange(0.1, 1.0, 0.1),
        "sigma_blurr": np.arange(0.1, 1.0, 0.1),
        "size_blurr": [10],
        "name": SOLVERS.keys(),
        "n_atoms": [50, 100],
        "lambd": [0.01, 0.05, 0.1],
        "dim_atoms": [10],
        "n_iter": [1000],
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
