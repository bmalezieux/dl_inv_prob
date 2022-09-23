
import numpy as np
import os
import torch
import torch.nn.functional as F
import pandas as pd
import itertools

from dl_inv_prob.common_utils import (
    pil_to_np,
)
from dl_inv_prob.dl import Deconvolution
from dl_inv_prob.utils import gaussian_kernel, psnr, is_divergence, discrepancy_measure
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from joblib import Memory

N_EXP = 20
EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results")
DATA = os.path.join(EXPERIMENTS, "data")
IMG = os.path.join(DATA, "flowers.png")
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
RESULT_FILE = "deblurring_cdl_supervised_n_atoms.csv"


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

SIZE = 256

mem = Memory(location="./tmp_deblurring_cdl_supervised_n_atoms/", verbose=0)


###################
# Data generation #
###################


def generate_data(img_path, size_kernel, sigma_blurr, size_image):

    img = Image.open(img_path).convert("L").resize((size_image, size_image),
                                                   Image.ANTIALIAS)
    img = pil_to_np(img)

    y = np.array(img)
    y = torch.tensor(y, device=DEVICE, dtype=torch.float, requires_grad=False)
    kernel = gaussian_kernel(size_kernel, sigma_blurr)
    kernel = torch.tensor(kernel, device=DEVICE,
                          dtype=torch.float,
                          requires_grad=False)
    y_conv = F.conv_transpose2d(y[None, :, :], kernel[None, None, :, :])
    y_conv_display = F.conv2d(
        y[None, :, :],
        torch.flip(kernel[None, None, :, :], dims=[2, 3]),
        padding="same"
    )

    y_conv = y_conv.detach().cpu().numpy().squeeze()
    y_conv_display = y_conv_display.detach().cpu().numpy().squeeze()
    kernel = kernel.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()

    return y, y_conv, y_conv_display, kernel


#############
# Benchmark #
#############


@mem.cache
def run_test(params):
    """
    Runs several tests using a given set of parameters and a given solver
    """

    y, y_conv, y_conv_display, kernel = generate_data(
        params["img_path"],
        params["size_kernel"],
        params["sigma_blurr"],
        params["size_image"]
    ) 

    psnr_recs = []
    is_recs = []
    avg_discrepancies = []
    avg_discrepancies_weighted = []

    for _ in range(N_EXP):

        # Dictionary learning supervised
        cdl = Deconvolution(
            params["n_atoms"],
            init_D=None,
            device=DEVICE,
            rng=NP_RNG,
            atom_height=params["dim_atoms"],
            atom_width=params["dim_atoms"],
            lambd=params["lambd"]
        )
        cdl.fit(y[None, :, :], np.array([1])[None, None, None, :])
        out = cdl.forward(cdl.Y_tensor)

        # Reconstruction
        rec = cdl.rec(y_conv[None, :, :], kernel[None, None, :, :]).squeeze()
        rec = np.clip(rec, 0, 1)

        weights = torch.abs(out).sum(axis=(2, 3)).to("cpu").detach().numpy()[0]
        D = cdl.D_
        discrepancy_mean = 0
        discrepancy_mean_weighted = 0
        for i in range(D.shape[0]):
            discrepancy_mean += discrepancy_measure(D[i, 0])
            discrepancy_mean_weighted += discrepancy_measure(D[i, 0]) * weights[i]
        discrepancy_mean /= D.shape[0]
        discrepancy_mean_weighted /= weights.sum()

        # Result
        psnr_rec = psnr(rec, y)
        is_rec = is_divergence(rec, y)

        psnr_recs.append(psnr_rec)
        is_recs.append(is_rec)
        avg_discrepancies.append(discrepancy_mean)
        if not np.isnan(discrepancy_mean_weighted):
            avg_discrepancies_weighted.append(discrepancy_mean_weighted)

    results = {
        "psnr_rec": np.mean(psnr_recs),
        "is_rec": np.mean(is_recs),
        "discrepancy": np.mean(avg_discrepancies),
        "discrepancy_weighted": np.mean(avg_discrepancies_weighted)
    }

    return results


###############
# Main script #
###############


if __name__ == "__main__":

    hyperparams = {
        "n_atoms": np.linspace(2, 20, num=10, dtype=int),
        "lambd": [1.],
        "dim_atoms": [20],
        "n_iter": [100],
        "sigma_blurr": [0.3],
        "size_kernel": [10],
        "img_path": [IMG],
        "size_image": [256]
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dict_results = {}

    for params in tqdm(permuts_params):
        try:
            results = run_test(params)

            # Storing results

            for key in params.keys():
                if key != "img_path":
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
