from dl_inv_prob.dl import Inpainting
from dl_inv_prob.utils import (create_patches_overlap, generate_dico,
                               patch_average, psnr, recovery_score)
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


DATA_PATH = '../data/flowers.png'
DEVICE = "cuda:3"
N_EXP = 10
RNG = np.random.default_rng(2022)

dim_image = 128
dim_patch = 8
patch_len = dim_patch ** 2
n_atoms = 100
s_values = np.arange(0.1, 1.0, 0.1)

# Image preprocessing
img = Image.open(DATA_PATH)
img = np.array(img.resize((dim_image,
                           dim_image), Image.ANTIALIAS).convert('L')) / 255.

# Patch extraction
y, _ = create_patches_overlap(img, dim_patch)
trivial_masks = np.ones(y.shape)

dict_results = {
    "scores": [],
    "scores_weights": [],
    "psnrs": [],
    "psnrs_corrupted": [],
    "s_values": []
}

for n_exp in range(N_EXP):
    # Shared random initialization
    D_init = generate_dico(n_atoms, patch_len, rng=RNG)

    # Dictionary learning without inpainting (random initialization)
    dl = Inpainting(
        n_atoms,
        init_D=D_init,
        device=DEVICE,
        rng=RNG
    )
    dl.fit(y[:, :, None], trivial_masks[:, :, None])
    D_no_inpainting = dl.D_

    for i, s in tqdm(enumerate(s_values)):
        # Binary masking with s (in proportion) missing values
        A = RNG.binomial(1, 1 - s, size=(dim_image, dim_image))
        img_inpainting = img * A
        y_inpainting, masks = create_patches_overlap(img_inpainting,
                                                     dim_patch, A)

        # Inpainting with overlapping patches (same patch initialization)
        dl_inpainting = Inpainting(
            n_atoms,
            init_D=D_init,
            device=DEVICE,
            rng=RNG
        )
        dl_inpainting.fit(y_inpainting[:, :, None], masks[:, :, None])
        D_inpainting = dl_inpainting.D_

        Y_tensor = torch.from_numpy(y_inpainting[:, :, None]).float().to(dl_inpainting.device)
        with torch.no_grad():
            codes = dl_inpainting.forward(Y_tensor).detach().to("cpu").numpy()

        weights = np.abs(codes).sum(axis=(0, 2))

        # Compute the reconstructed image
        rec_patches = dl_inpainting.rec(y_inpainting[:, :, None])
        rec = patch_average(rec_patches, dim_patch,
                            dim_image, dim_image)
        rec = np.clip(rec, 0, 1)

        dict_results["scores"].append(recovery_score(D_inpainting, D_no_inpainting))
        dict_results["scores_weights"].append(recovery_score(D_inpainting, D_no_inpainting, weights))
        dict_results["psnrs"].append(psnr(rec, img))
        dict_results["psnrs_corrupted"].append(psnr(img_inpainting, img))
        dict_results["s_values"].append(s)


results_df = pd.DataFrame(dict_results)
results_df.to_csv("../results/inpainting_patches.csv")
