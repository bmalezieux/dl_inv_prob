from dl_inv_prob.dl import DictionaryLearning, Inpainting
from dl_inv_prob.utils import (extract_patches, combine_patches, psnr,
                               create_patches_overlap, patch_average,
                               generate_dico, recovery_score)
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


DATA_PATH = '../data/flowers.png'
DEVICE = "cuda:2"
N_EXP = 1
RNG = np.random.default_rng(2018)

dim_image = 200
dim_patch = 10
patch_len = dim_patch ** 2
n_patches = (dim_image // dim_patch) ** 2
n_atoms = 100
nb_s = 5
s_values = np.linspace(0, 1, nb_s)

scores = np.zeros((N_EXP, nb_s))
psnrs = np.zeros((N_EXP, nb_s))
psnrs_patch_init = np.zeros((N_EXP, nb_s))
psnrs_corrupted = np.zeros((N_EXP, nb_s))
psnrs_overlap = np.zeros((N_EXP, nb_s))

rec_images = np.zeros((N_EXP, nb_s, dim_image, dim_image))
rec_images_patch_init = np.zeros((N_EXP, nb_s, dim_image, dim_image))
corrupted_images = np.zeros((N_EXP, nb_s, dim_image, dim_image))
rec_images_overlap = np.zeros((N_EXP, nb_s, dim_image, dim_image))

# Image preprocessing
img = Image.open(DATA_PATH)
img = np.array(img.resize((dim_image,
                           dim_image), Image.ANTIALIAS).convert('L')) / 255
patches = extract_patches(img, dim_patch)
y = patches.reshape((n_patches, patch_len))

for n_exp in tqdm(range(N_EXP)):
    # Shared random initialization
    D_init = generate_dico(n_atoms, patch_len, rng=RNG)

    # Dictionary learning without inpainting (random initialization)
    dl = DictionaryLearning(
        n_atoms,
        init_D=D_init,
        device=DEVICE,
        rng=RNG
    )
    dl.fit(y[:, :, None])
    D_no_inpainting = dl.D_

    for i, s in tqdm(enumerate(s_values)):
        # Binary masking with s (in proportion) missing values
        A = RNG.binomial(1, 1 - s, size=(dim_image, dim_image))
        masks = extract_patches(A, dim_patch).reshape((n_patches, patch_len))
        y_inpainting = masks * y
        y_overlap, masks_overlap = create_patches_overlap(img * A,
                                                          dim_patch, A)

        # Dictionary learning with inpainting (random initialization)
        dl_inpainting = Inpainting(
            n_atoms,
            init_D=D_init,
            device=DEVICE,
            rng=RNG
        )
        dl_inpainting.fit(y_inpainting[:, :, None], masks)
        D_inpainting = dl_inpainting.D_

        scores[n_exp, i] = recovery_score(D_inpainting, D_no_inpainting)

        # Inpainting with overlapping patches
        dl_overlap = Inpainting(
            n_atoms,
            init_D=D_init,
            device=DEVICE,
            rng=RNG
        )
        dl_overlap.fit(y_overlap[:, :, None], masks_overlap)
        D_overlap = dl_overlap.D_

        # Initialization with corrupted random patches
        indices = RNG.choice(y_overlap.shape[0],
                             n_atoms, replace=False)
        D_init_patch_inpainting = y_overlap[indices, :]

        # Dictionary learning with inpainting (patch initialization)
        dl_inpainting_patch_init = Inpainting(
            n_atoms,
            init_D=D_init_patch_inpainting,
            device=DEVICE,
            rng=RNG
        )
        dl_inpainting_patch_init.fit(y_inpainting[:, :, None], masks)
        D_inpainting_patch_init = dl_inpainting_patch_init.D_

        # Compute the reconstructed image for each method
        rec_patches = dl_inpainting.rec(y_inpainting[:, :, None])
        rec_patches = rec_patches.reshape((n_patches, dim_patch, dim_patch))
        rec_inpainting = combine_patches(rec_patches)
        rec_inpainting = np.clip(rec_inpainting, 0, 1)

        rec_patches_patch_init = dl_inpainting_patch_init.rec(
            y_inpainting[:, :, None])
        rec_patches_patch_init = rec_patches_patch_init.reshape((n_patches,
                                                                 dim_patch,
                                                                 dim_patch))
        rec_inpainting_patch_init = combine_patches(rec_patches_patch_init)
        rec_inpainting_patch_init = np.clip(rec_inpainting_patch_init, 0, 1)

        rec_patches_overlap = dl_overlap.rec(y_overlap[:, :, None])
        rec_overlap = patch_average(rec_patches_overlap, dim_patch,
                                    dim_image, dim_image)
        rec_overlap = np.clip(rec_overlap, 0, 1)

        # Store images
        patches_inpainting = y_inpainting.reshape((n_patches,
                                                  dim_patch,
                                                  dim_patch))
        img_inpainting = combine_patches(patches_inpainting)

        rec_images[n_exp, i, :] = rec_inpainting
        rec_images_patch_init[n_exp, i, :] = rec_inpainting_patch_init
        corrupted_images[n_exp, i, :] = img_inpainting
        rec_images_overlap[n_exp, i, :] = rec_overlap

        # Store psnrs
        psnrs[n_exp, i] = psnr(rec_inpainting, img)
        psnrs_patch_init[n_exp, i] = psnr(rec_inpainting_patch_init, img)
        psnrs_corrupted[n_exp, i] = psnr(img_inpainting, img)
        psnrs_overlap[n_exp, i] = psnr(rec_overlap, img)

results_df = {
    "scores": {"scores": scores},
    "psnrs": {"psnrs": psnrs},
    "psnrs_patch_init": {"psnrs_patch_init": psnrs_patch_init},
    "psnrs_corrupted": {"psnrs_corrupted": psnrs_corrupted},
    "psnrs_overlap": {"psnrs_overlap": psnrs_overlap},
    "s_values": {"s_values": s_values}
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("../results/inpainting_patches.pickle")

np.save("../results/inpainting_patches_rec_images.npy", rec_images)
np.save("../results/inpainting_patches_rec_images_patch_init.npy",
        rec_images_patch_init)
np.save("../results/inpainting_patches_corrupted_images.npy", corrupted_images)
np.save("../results/inpainting_patches_rec_images_overlap.npy",
        rec_images_overlap)
