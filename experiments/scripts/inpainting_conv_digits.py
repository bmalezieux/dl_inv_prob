from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.utils import create_image_digits, rec_score_digits
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm


N_EXP = 1
DEVICE = 'cuda:1'
RNG = np.random.default_rng(2022)

n_atoms = 100
dim_atom = 12

sizes = np.arange(100, 501, 100)
n_sizes = sizes.shape[0]
s_values = np.linspace(0.1, 0.5, 5)
n_s = s_values.shape[0]
scores = np.zeros((n_s, n_sizes))

D_ref = load_digits().images[:10]
D_ref /= np.linalg.norm(D_ref, axis=(1, 2), keepdims=True)

for exp in tqdm(range(N_EXP)):
    for i, size in tqdm(enumerate(sizes)):
        for j, s in tqdm(enumerate(s_values)):

            y = create_image_digits(size, size, k=0.1, rng=RNG)
            mask = RNG.binomial(1, 1 - s, y.shape)
            y_inpainting = y * mask
            patches = extract_patches_2d(y_inpainting, (dim_atom, dim_atom),
                                         max_patches=n_atoms,
                                         random_state=RNG.integers(1000))
            D_init = patches.reshape((n_atoms, 1, dim_atom, dim_atom))
            scores_lambd = np.zeros(3)

            for k, lambd in enumerate([0.1, 0.2, 0.3]):
                dl = ConvolutionalInpainting(
                    n_atoms,
                    lambd=lambd,
                    init_D=D_init,
                    atom_height=dim_atom,
                    atom_width=dim_atom,
                    device=DEVICE,
                    rng=RNG
                )
                dl.fit(y_inpainting[None, :, :], mask)
                D = dl.D_.squeeze()
                scores_lambd[k] = rec_score_digits(D, D_ref)

            scores[exp, j, i] = scores_lambd.max()

results_df = {
    "scores": {"scores": scores},
    "sizes": {"sizes": sizes},
    "s_values": {"s_values": s_values}
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("../results/inpainting_conv_digits.pickle")
