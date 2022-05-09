from dl_inv_prob.dl import DictionaryLearning, Inpainting
from dl_inv_prob.utils import (extract_patches,
                               generate_dico,
                               recovery_score)
import numpy as np
from PIL import Image


DATA_PATH = '../data/flowers.png'
DEVICE = "cuda:1"
N_EXP = 1
RNG = np.random.default_rng(2022)

dim_image = 200
dim_patch = 10
patch_len = dim_patch ** 2
n_patches = (dim_image // dim_patch) ** 2
n_atoms = 100
nb_s = 2
s_values = np.linspace(0, 1, nb_s)

scores = np.zeros((N_EXP, nb_s))

# Image preprocessing
img = Image.open(DATA_PATH)
img = np.array(img.resize((200, 200), Image.ANTIALIAS).convert('L')) / 255
patches = extract_patches(img, dim_patch)
y = patches.reshape((n_patches, patch_len))

for n_exp in range(N_EXP):
    # Random dictionary initialization
    D_init = generate_dico(n_atoms, patch_len, rng=RNG)

    # Dictionary learning without inpainting (random initialization)
    dl = DictionaryLearning(
        n_atoms,
        init_D=D_init.copy(),
        device=DEVICE,
        rng=RNG
    )
    dl.fit(y[:, :, None])
    D_no_inpainting = dl.D_

    # Trivial binary mask
    masks = np.ones((n_patches, patch_len))
    A = np.concatenate([np.diag(mask)[None, :] for mask in masks])

    # Dictionary learning with trivial inpainting (same initialization)
    dl_inpainting = Inpainting(
        n_atoms,
        init_D=D_init.copy(),
        device=DEVICE,
        rng=RNG
    )
    dl_inpainting.fit(y[:, :, None], masks)
    D_inpainting = dl_inpainting.D_

    # Dictionary learning with trivial A (same initialization)
    dl_A = DictionaryLearning(
        n_atoms,
        init_D=D_init.copy(),
        device=DEVICE,
        rng=RNG
    )
    dl_A.fit(y[:, :, None], A)
    D_A = dl_A.D_

    print(recovery_score(D_inpainting, D_no_inpainting),
          recovery_score(D_A, D_no_inpainting))
