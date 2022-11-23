#%%

import datetime
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F


from dl_inv_prob.common_utils import pil_to_np, torch_to_np
from dl_inv_prob.dl import Deconvolution
from dl_inv_prob.utils import gaussian_kernel, psnr, determinist_blurr
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

sigma_blurr = 0.3

IMGS = ["flowers", "forest", "animal", "mushroom"]

for IMG in IMGS:
    IMG_PATH = os.path.join(DATA, f"{IMG}.png")

    # img = (
    #     Image.open(IMG_PATH).convert("L").resize((SIZE, SIZE), Image.ANTIALIAS)
    # )
    # img = pil_to_np(img)

    # Data generation
    # img = torch.tensor(
    #     img, device=DEVICE, dtype=torch.float, requires_grad=False
    # )
    # kernel = gaussian_kernel(10, sigma_blurr)
    # kernel = torch.tensor(
    #     kernel, device=DEVICE, dtype=torch.float, requires_grad=False
    # )
    # img_blurr = F.conv_transpose2d(img[None, :, :], kernel[None, None, :, :])
    # img_blurr_display = F.conv2d(
    #     img[None, :, :],
    #     torch.flip(kernel[None, None, :, :], dims=[1, 2]),
    #     padding="same",
    # )

    # img_blurr = img_blurr.detach().cpu().numpy().squeeze()
    # img_blurr_display = img_blurr_display.detach().cpu().numpy().squeeze()
    # kernel = kernel.detach().cpu().numpy().squeeze()
    # img = img.detach().cpu().numpy().squeeze()

    # img_blurr = np.clip(img_blurr, 0, 1)
    # img_blurr_display = np.clip(img_blurr_display, 0, 1)

    img, img_blurr, kernel = determinist_blurr(
        IMG_PATH, sigma_blurr, 10, sigma=0.0, size=SIZE
    )
    img = torch_to_np(img).squeeze()
    img_blurr = torch_to_np(img_blurr).squeeze()
    kernel = kernel.numpy()

    # CDL
    n_atoms = 50
    dim_atoms = 20
    lambd = 0.1

    cdl = Deconvolution(
        n_components=n_atoms,
        lambd=lambd,
        atom_height=dim_atoms,
        atom_width=dim_atoms,
        device=DEVICE,
        rng=NP_RNG,
    )

    start = time.time()

    cdl.fit(img_blurr[None, :, :], kernel[None, :, :])
    rec = cdl.rec().squeeze()
    rec = np.clip(rec, 0, 1)

    stop = time.time()
    delta = stop - start
    delta = str(datetime.timedelta(seconds=delta))
    print(f"time: {delta}")

    #%%

    D = cdl.D.cpu().detach().numpy().squeeze()
    img_blurr_torch = torch.from_numpy(img_blurr).float().to(cdl.device)
    z = (
        cdl.forward(img_blurr_torch[None, :, :])
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )

    weights = np.sum(np.abs(z), axis=(1, 2))
    idxs = np.argsort(weights)[::-1]
    D = D[idxs, :, :]

    np.save(os.path.join(RESULTS, f"deblurring_dict_{IMG}.npy"), D)

    # Result
    psnr_rec = psnr(rec, img)

    print(f"psnr_rec = {psnr_rec:.2f}\n")

    img_blurr *= 255
    img_blurr = Image.fromarray(img_blurr.squeeze().astype(np.uint8))
    img_blurr.save(os.path.join(RESULTS, f"blurred_{IMG}.png"))

    rec *= 255
    rec = Image.fromarray(rec.squeeze().astype(np.uint8))
    rec.save(os.path.join(RESULTS, f"rec_deblurring_{IMG}.png"))

    img *= 255
    img = Image.fromarray(img.squeeze().astype(np.uint8))
    img.save(os.path.join(RESULTS, f"clean_{IMG}.png"))
