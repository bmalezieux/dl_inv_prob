from dl_inv_prob.dl import Inpainting
from dl_inv_prob.utils import (create_patches_overlap, generate_dico,
                               patch_average, psnr, recovery_score)
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


DATA_PATH = '../data/flowers.png'
DEVICE = "cuda:3"
N_EXP = 1
RNG = np.random.default_rng(2022)

dim_image = 200
dim_patch = 10
patch_len = dim_patch ** 2
n_atoms = 100
nb_s = 5
s_values = np.linspace(0, 1, nb_s)

scores = np.zeros((N_EXP, nb_s))
psnrs = np.zeros((N_EXP, nb_s))
psnrs_corrupted = np.zeros((N_EXP, nb_s))

rec_images = np.zeros((N_EXP, nb_s, dim_image, dim_image))
corrupted_images = np.zeros((N_EXP, nb_s, dim_image, dim_image))

# Image preprocessing
img = Image.open(DATA_PATH)
img = np.array(img.resize((dim_image,
                           dim_image), Image.ANTIALIAS).convert('L')) / 255

# Patch extraction
y, _ = create_patches_overlap(img, dim_patch)
trivial_masks = np.ones(y.shape)

for n_exp in tqdm(range(N_EXP)):
    # Shared random initialization
    D_init = generate_dico(n_atoms, patch_len, rng=RNG)

    # Dictionary learning without inpainting (random initialization)
    dl = Inpainting(
        n_atoms,
        init_D=D_init,
        device=DEVICE,
        rng=RNG
    )
    dl.fit(y[:, :, None], trivial_masks)
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
        dl_inpainting.fit(y_inpainting[:, :, None], masks)
        D_inpainting = dl_inpainting.D_

        scores[n_exp, i] = recovery_score(D_inpainting, D_no_inpainting)

        # Compute the reconstructed image for each method
        rec_patches = dl_inpainting.rec(y_inpainting[:, :, None])
        rec = patch_average(rec_patches, dim_patch,
                            dim_image, dim_image)
        rec = np.clip(rec, 0, 1)

        # Store images
        rec_images[n_exp, i, :] = rec
        corrupted_images[n_exp, i, :] = img_inpainting

        # Store psnrs
        psnrs[n_exp, i] = psnr(rec, img)
        psnrs_corrupted[n_exp, i] = psnr(img_inpainting, img)

results_df = {
    "scores": {"scores": scores},
    "psnrs": {"psnrs": psnrs},
    "psnrs_corrupted": {"psnrs_corrupted": psnrs_corrupted},
    "s_values": {"s_values": s_values}
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("../results/inpainting_patches.pickle")

np.save("../results/inpainting_patches_rec_images.npy", rec_images)
np.save("../results/inpainting_patches_corrupted_images.npy", corrupted_images)
