#%%

import datetime
import numpy as np
import os
import random
import time
import torch

from dl_inv_prob.common_utils import (
    pil_to_np,
)
from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.utils import psnr
from pathlib import Path
from PIL import Image

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = os.path.join(EXPERIMENTS, "data")
RESULTS = os.path.join(EXPERIMENTS, "results", "dictionaries")
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

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

SIZE = 256

os.makedirs(RESULTS, exist_ok=True)

sigma = 0.0
rho = 0.5

IMGS = ["flowers", "forest", "animal", "mushroom"]

for IMG in IMGS:
    IMG_PATH = os.path.join(DATA, f"{IMG}.png")

    img = (
        Image.open(IMG_PATH).convert("L").resize((SIZE, SIZE), Image.ANTIALIAS)
    )
    img = pil_to_np(img)

    # Data generation
    mask = NP_RNG.binomial(1, rho, size=img.shape)
    noise = NP_RNG.normal(0, sigma, size=img.shape)
    img_inpainting = (img + noise) * mask
    img_inpainting = np.clip(img_inpainting, 0, 1)

    # CDL
    n_atoms = 50
    dim_atoms = 20
    lambd = 0.1

    cdl = ConvolutionalInpainting(
        n_components=n_atoms,
        lambd=lambd,
        atom_height=dim_atoms,
        atom_width=dim_atoms,
        device=DEVICE,
        rng=NP_RNG,
    )

    start = time.time()

    cdl.fit(img_inpainting, mask)
    rec = cdl.rec().squeeze()
    rec = np.clip(rec, 0, 1)

    stop = time.time()
    delta = stop - start
    delta = str(datetime.timedelta(seconds=delta))
    print(f"time: {delta}")

    #%%

    D = cdl.D.cpu().detach().numpy().squeeze()
    img_inpainting_torch = (
        torch.from_numpy(img_inpainting).float().to(cdl.device)
    )
    z = cdl.forward(img_inpainting_torch).cpu().detach().numpy().squeeze()

    weights = np.sum(np.abs(z), axis=(1, 2))
    idxs = np.argsort(weights)[::-1]
    D = D[idxs, :, :]

    np.save(os.path.join(RESULTS, f"inpainting_dict_{IMG}.npy"), D)

    # Result
    psnr_rec = psnr(rec, img)
    psnr_corr = psnr(img_inpainting, img)

    print(f"psnr_rec = {psnr_rec:.2f}, psnr_corr = {psnr_corr:.2f}\n")

    img_inpainting *= 255
    img_inpainting = Image.fromarray(img_inpainting.squeeze().astype(np.uint8))
    img_inpainting.save(os.path.join(RESULTS, f"inpainted_{IMG}.png"))

    rec *= 255
    rec = Image.fromarray(rec.squeeze().astype(np.uint8))
    rec.save(os.path.join(RESULTS, f"rec_inpainting_{IMG}.png"))

    img *= 255
    img = Image.fromarray(img.squeeze().astype(np.uint8))
    img.save(os.path.join(RESULTS, f"clean_{IMG}.png"))
