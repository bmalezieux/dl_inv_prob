from dl_inv_prob.dl import DictionaryLearning
from dl_inv_prob.utils import (extract_patches, combine_patches,
                               psnr, recovery_score)
import numpy as np
from PIL import Image
from tqdm import tqdm


DEVICE = "cuda:1"
N_EXP = 10
RNG = np.random.default_rng(100)

dim_image = 200
dim_patch = 10
patch_len = dim_patch ** 2
n_patches = (dim_image // dim_patch) ** 2

n_atoms = 100
nb_s = 10
s_values = np.linspace(0, 1, nb_s)

img = Image.open('../data/flowers.png')
img = np.array(img.resize((200, 200)).convert('L'))
patches = extract_patches(img, dim_patch)
y = patches.reshape((n_patches, patch_len))

scores = np.zeros((N_EXP, nb_s))
psnrs = np.zeros((N_EXP, nb_s))

dl = DictionaryLearning(
    n_atoms,
    device=DEVICE,
    rng=RNG
)
dl.fit(y)
D_no_inpainting = dl.D_

for n_exp in tqdm(range(N_EXP)):

    for i, s in enumerate(s_values):

        mask = RNG.random.binomial(1, s, size=(n_patches, patch_len))
        y_inpainting = mask * y

        dl_inpainting = DictionaryLearning(
            n_atoms,
            device=DEVICE,
            rng=RNG
        )
        dl_inpainting.fit(y_inpainting, mask)
        D_inpainting = dl_inpainting.D_
        
        rec_patches = dl_inpainting.rec().reshape((n_patches, 
                                                   dim_patch, 
                                                   dim_patch))
        rec_inpainting = combine_patches(rec_patches)

        scores[n_exp, i] = recovery_score(D_inpainting, D_no_inpainting)
        psnrs[n_exp, i] = psnr(rec_inpainting, img)


np.save("../results/inpainting_patches_scores.npy", scores)
np.save("../results/inpainting_patches_psnrs.npy", psnrs)
np.save("../results/inpainting_patches_s_values.npy", s_values)
