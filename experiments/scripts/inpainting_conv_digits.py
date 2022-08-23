from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.utils import create_image_digits, rec_score_digits
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from tqdm import tqdm

N_EXP = 1
DEVICE = 'cuda:1'
RNG = np.random.default_rng(2022)

n_atoms = 100
dim_atom = 10

sizes = np.arange(100, 801, 100)
n_sizes = sizes.shape[0]
s_values = np.array([0, 0.1, 0.3, 0.4, 0.5])
n_s = s_values.shape[0]
lambds = [0.1, 0.2, 0.3]
scores = np.zeros((N_EXP, n_s, n_sizes))

D_init = RNG.normal(size=(n_atoms, 1, dim_atom, dim_atom))
D_ref = load_digits().images[:10]
D_ref /= np.linalg.norm(D_ref, axis=(1, 2), keepdims=True)

for exp in tqdm(range(N_EXP)):
    print(f'exp {exp}')
    for i, size in enumerate(sizes):
        print(f'size {size}')
        for j, s in enumerate(s_values):
            print(f's {s:.2f}')

            y = create_image_digits(size, size, k=0.1, rng=RNG)
            mask = RNG.binomial(1, 1 - s, y.shape)
            y_inpainting = y * mask

            scores_lambd = np.zeros(len(lambds))

            for k, lambd in enumerate(lambds):
                dl = ConvolutionalInpainting(
                    n_atoms,
                    lambd=lambd,
                    init_D=D_init,
                    atom_height=dim_atom,
                    atom_width=dim_atom,
                    device=DEVICE,
                    rng=RNG,
                    alpha=0.1
                )
                dl.fit(y_inpainting[None, :, :], mask)
                D = dl.D_.squeeze()
                scores_lambd[k] = rec_score_digits(D, D_ref)

            scores[exp, j, i] = scores_lambd.max()
            print(f'score = {scores[exp, j, i]}')

results_df = {
    "scores": {"scores": scores},
    "sizes": {"sizes": sizes},
    "s_values": {"s_values": s_values}
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("../results/inpainting_conv_digits.pickle")
