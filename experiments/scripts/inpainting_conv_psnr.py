from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.utils import psnr
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


DATA_PATH = '../data/flowers.png'
DEVICE = "cuda:3"
N_EXP = 1
RNG = np.random.default_rng(2022)

dim_image = 200
dim_atom = 10
n_atoms = 100
nb_s = 5
s_values = np.linspace(0, 1, nb_s)

psnrs = np.zeros((N_EXP, nb_s))
psnrs_corrupted = np.zeros((N_EXP, nb_s))

rec_images = np.zeros((N_EXP, nb_s, dim_image, dim_image))
corrupted_images = np.zeros((N_EXP, nb_s, dim_image, dim_image))

# Image preprocessing
img = Image.open(DATA_PATH)
y = np.array(img.resize((dim_image,
                         dim_image), Image.ANTIALIAS).convert('L')) / 255

for n_exp in tqdm(range(N_EXP)):
    # Shared random initialization
    D_init = RNG.normal(size=(n_atoms, 1, dim_atom, dim_atom))

    for i, s in tqdm(enumerate(s_values)):
        # Binary masking with s (in proportion) missing values
        mask = RNG.binomial(1, 1 - s, size=(dim_image, dim_image))
        y_inpainting = y * mask

        # Inpainting with overlapping patches (same patch initialization)
        dl = ConvolutionalInpainting(
            n_atoms,
            init_D=D_init,
            atom_height=dim_atom,
            atom_width=dim_atom,
            device=DEVICE,
            rng=RNG
        )
        dl.fit(y_inpainting[None, :, :], mask)
        D = dl.D_

        # Compute the reconstructed image
        rec = dl.rec(y_inpainting[None, :, :]).squeeze()
        rec = np.clip(rec, 0, 1)

        # Store images
        rec_images[n_exp, i, :] = rec
        corrupted_images[n_exp, i, :] = y_inpainting

        # Store psnrs
        psnrs[n_exp, i] = psnr(rec, y)
        psnrs_corrupted[n_exp, i] = psnr(y_inpainting, y)

results_df = {
    "psnrs": {"psnrs": psnrs},
    "psnrs_corrupted": {"psnrs_corrupted": psnrs_corrupted},
    "s_values": {"s_values": s_values}
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("../results/inpainting_conv.pickle")

np.save("../results/inpainting_conv_rec_images.npy", rec_images)
np.save("../results/inpainting_conv_corrupted_images.npy", corrupted_images)
