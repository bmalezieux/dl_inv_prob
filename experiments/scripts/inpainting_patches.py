import json
import jsonlines
import os

from dl_inv_prob.dl import Inpainting
from dl_inv_prob.utils import (create_patches_overlap, generate_dico,
                               patch_average, psnr, recovery_score)
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Paths and constants
DATA_PATH = Path(__file__).parent / '../data/flowers.png'
RESULTS_PATH = Path(__file__).parent / "../results/"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
JSONL_PATH = RESULTS_PATH / "inpainting_patches.jsonl"
CACHE_PATH = RESULTS_PATH / "inpainting_cache.jsonl"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_EXP = 5
RNG = np.random.default_rng(2022)
dims_image = [128, 256, 502]
dim_patch = 10
patch_len = dim_patch ** 2
n_atoms = 100
s_values = np.arange(0.1, 1.0, 0.1)

# Load cache if exists
completed = set()
if CACHE_PATH.exists():
    with jsonlines.open(CACHE_PATH, mode='r') as reader:
        for record in reader:
            completed.add((record["dim_image"], record["n_exp"], record["s"]))

# Main processing
img_orig = Image.open(DATA_PATH).convert('L')

with jsonlines.open(JSONL_PATH, mode='a') as writer, \
     jsonlines.open(CACHE_PATH, mode='a') as cache_writer:

    for dim_image in tqdm(dims_image, desc="Image Sizes"):
        img = np.array(img_orig.resize((dim_image, dim_image))) / 255.
        y, _ = create_patches_overlap(img, dim_patch)
        trivial_masks = np.ones(y.shape)

        for n_exp in tqdm(range(N_EXP), desc=f"Experiments for {dim_image}", leave=False):
            D_init = generate_dico(n_atoms, patch_len, rng=RNG)
            dl = Inpainting(n_atoms, init_D=D_init, device=DEVICE, rng=RNG, max_iter=50)
            dl = torch.compile(dl)
            dl.fit(y[:, :, None], trivial_masks[:, :, None])
            D_no_inpainting = dl.D_

            for s in tqdm(s_values, desc="Sparsity", leave=False):
                key = (dim_image, n_exp, float(s))
                if key in completed:
                    continue

                A = RNG.binomial(1, 1 - s, size=(dim_image, dim_image))
                img_inpainting = img * A
                y_inpainting, masks = create_patches_overlap(img_inpainting, dim_patch, A)

                dl_inpainting = Inpainting(n_atoms, init_D=D_init, device=DEVICE, rng=RNG)
                dl_inpainting = torch.compile(dl_inpainting)
                dl_inpainting.fit(y_inpainting[:, :, None], masks[:, :, None])
                D_inpainting = dl_inpainting.D_

                Y_tensor = torch.from_numpy(y_inpainting[:, :, None]).float().to(DEVICE)
                with torch.no_grad():
                    codes = dl_inpainting.forward(Y_tensor).detach().cpu().numpy()
                weights = np.abs(codes).sum(axis=(0, 2))

                rec_patches = dl_inpainting.rec(y_inpainting[:, :, None])
                rec = patch_average(rec_patches, dim_patch, dim_image, dim_image)
                rec = np.clip(rec, 0, 1)

                result = {
                    "dim_image": dim_image,
                    "n_exp": n_exp,
                    "s": float(s),
                    "score": recovery_score(D_inpainting, D_no_inpainting),
                    "score_weighted": recovery_score(D_inpainting, D_no_inpainting, weights),
                    "psnr": psnr(rec, img),
                    "psnr_corrupted": psnr(img_inpainting, img)
                }

                writer.write(result)
                cache_writer.write(result)  # log to cache
