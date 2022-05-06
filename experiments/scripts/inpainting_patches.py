from mimetypes import init
from dl_inv_prob.dl import DictionaryLearning
from dl_inv_prob.utils import (extract_patches, combine_patches,
                               generate_dico, init_dictionary_img,
                               psnr, recovery_score)
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


DATA_PATH = '../data/flowers.png'
DEVICE = "cuda:1"
N_EXP = 1
RNG = np.random.default_rng(100)

dim_image = 200
dim_patch = 10
patch_len = dim_patch ** 2
n_patches = (dim_image // dim_patch) ** 2

n_atoms = 100
nb_s = 5
s_values = np.linspace(0, 1, nb_s)

# Image preprocessing
img = Image.open(DATA_PATH)
img = np.array(img.resize((200, 200), Image.ANTIALIAS).convert('L')) / 255
patches = extract_patches(img, dim_patch)
y = patches.reshape((n_patches, patch_len, 1))

D_init = generate_dico(n_atoms, patch_len, rng=RNG)

# Dictionary learning without inpainting
# D_init = init_dictionary_img(img, dim_patch, n_atoms, RNG)
dl = DictionaryLearning(
    n_atoms,
    init_D=D_init,
    device=DEVICE,
    rng=RNG
)
dl.fit(y)
D_no_inpainting = dl.D_

scores = np.zeros((N_EXP, nb_s))
psnrs = np.zeros((N_EXP, nb_s))
rec_images = np.zeros((N_EXP, nb_s, dim_image, dim_image))

for n_exp in tqdm(range(N_EXP)):

    for i, s in tqdm(enumerate(s_values)):

        masks = RNG.binomial(1, 1 - s, size=(n_patches, patch_len))
        A = np.concatenate([np.diag(mask)[None, :] for mask in masks])
        y_inpainting = A @ y

        # img_inpainting = combine_patches(y_inpainting.squeeze())
        # D_init_inpainting = init_dictionary_img(img_inpainting, dim_patch,
                                                # n_atoms, RNG)
        dl_inpainting = DictionaryLearning(
            n_atoms,
            init_D=D_init,
            device=DEVICE,
            rng=RNG
        )
        dl_inpainting.fit(y_inpainting, A)
        D_inpainting = dl_inpainting.D_
        
        scores[n_exp, i] = recovery_score(D_inpainting, D_no_inpainting)
        
        rec_patches = dl_inpainting.rec(y_inpainting).reshape((n_patches, 
                                                   dim_patch, 
                                                   dim_patch))
        rec_inpainting = combine_patches(rec_patches)
        rec_inpainting = np.clip(rec_inpainting, 0, 1)

        rec_images[n_exp, i, :] = rec_inpainting
        psnrs[n_exp, i] = psnr(rec_inpainting, img)

results_df = {
    "scores": {"scores": scores},
    "psnrs": {"psnrs": psnrs},
    "s_values": {"s_values": s_values}
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("../results/inpainting_patches.pickle")

np.save("../results/inpainting_patches_rec_images.npy", rec_images)
