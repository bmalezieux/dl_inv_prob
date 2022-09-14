import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch

from dl_inv_prob.common_utils import (
    pil_to_np,
)
from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.dip import DIPInpainting
from dl_inv_prob.utils import psnr
from joblib import Memory
from pathlib import Path
from PIL import Image
from utils.tv import ProxTV
from utils.wavelets import SparseWavelets

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = os.path.join(EXPERIMENTS, "data")
IMG = os.path.join(DATA, "flowers.png")
RESULTS = os.path.join(EXPERIMENTS, "results", "inpainting_color")
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

mem = Memory(location="./tmp_inpainting_color/", verbose=0)

SIZE = 256

img = Image.open(IMG).convert("RGB").resize((SIZE, SIZE), Image.ANTIALIAS)
img = pil_to_np(img)

os.makedirs(RESULTS, exist_ok=True)
plt.imsave(os.path.join(RESULTS, "clean.png"), np.transpose(img, (1, 2, 0)))

# CDL
n_atoms = 50
dim_atoms = 10
lambd = 0.01

cdl = ConvolutionalInpainting(
    n_components=n_atoms,
    lambd=lambd,
    atom_height=dim_atoms,
    atom_width=dim_atoms,
    device=DEVICE,
    rng=NP_RNG,
)

# DIP
model = "SkipNet"
n_iter = 1000
lr = 0.01
sigma_input_noise = 0.1
sigma_reg_noise = 0.03
input_depth = 32

dip = DIPInpainting(
    model=model,
    n_iter=n_iter,
    lr=lr,
    sigma_input_noise=sigma_input_noise,
    sigma_reg_noise=sigma_reg_noise,
    input_depth=input_depth,
    output_depth=3,
    device=DEVICE,
)

# TV
lambd = 0.1
n_iter = 1000

tv = ProxTV(lambd=lambd, n_iter=n_iter)

# Wavelets
lambd = 0.01
n_iter = 1000
step = 1.0
wavelet = "db3"

wavelets = SparseWavelets(
    lambd=lambd, n_iter=n_iter, step=step, wavelet=wavelet
)


@mem.cache
def run_test(params):
    """
    Runs several tests using a given set of parameters and a given solver
    """

    rho = params["rho"]
    sigma = params["sigma"]
    solver = params["solver"]
    solver_name = params["solver_name"]

    print(f"solver: {solver_name}")
    print(f"rho = {rho}, sigma = {sigma}")

    # Reconstruction
    start = time.time()
    if solver_name == "dip":
        solver.fit(img_inpainting, mask)
        rec = solver.rec().squeeze()
    else:
        rec = np.zeros_like(img_inpainting)
        for i, channel in enumerate(img_inpainting):
            solver.fit(channel[None, :, :], mask[None, :, :])
            rec[i] = solver.rec().squeeze()
    rec = np.clip(rec, 0, 1)
    stop = time.time()

    # Result
    psnr_rec = psnr(rec, img)
    psnr_corr = psnr(img_inpainting, img)
    delta = stop - start
    delta = str(datetime.timedelta(seconds=delta))

    print(f"psnr_rec = {psnr_rec:.2f}, psnr_corr = {psnr_corr:.2f}")
    print(f"time: {delta}\n")

    return rec


sigmas = [0.0, 0.1]
rhos = [0.5]
solvers = [cdl, dip, tv, wavelets]
solver_names = ["cdl", "dip", "tv", "wavelets"]

dict_results = {}

for rho in rhos:
    for sigma in sigmas:

        # Data generation
        mask = NP_RNG.binomial(1, rho, size=img.shape[1:])
        noise = NP_RNG.normal(0, sigma, size=img.shape)
        img_inpainting = (img + noise) * mask
        img_inpainting = np.clip(img_inpainting, 0, 1)

        file_name = f"corrupted_{rho:.2f}_{sigma:.2f}.png"
        plt.imsave(
            os.path.join(RESULTS, file_name),
            np.transpose(img_inpainting, (1, 2, 0)),
        )

        for solver, solver_name in zip(solvers, solver_names):
            params = {
                "rho": rho,
                "sigma": sigma,
                "solver_name": solver_name,
                "solver": solver,
            }
            rec = run_test(params)

            file_name = f"{solver_name}_{rho:.2f}_{sigma:.2f}.png"
            plt.imsave(
                os.path.join(RESULTS, file_name),
                np.transpose(rec, (1, 2, 0)),
            )
