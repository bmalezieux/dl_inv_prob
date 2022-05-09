import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# plt.style.use('figures_style_small.mplstyle')

df = pd.read_pickle('../results/inpainting_patches.pickle')
psnrs = df.psnrs.psnrs
psnrs_patch_init = df.psnrs_patch_init.psnrs_patch_init
psnrs_corrupted = df.psnrs_corrupted.psnrs_corrupted
s_values = df.s_values.s_values
quantities = [psnrs, psnrs_patch_init, psnrs_corrupted]
labels = ['random init', 'patch init', 'corrupted image']

fig = plt.figure()

for quantity, label in zip(quantities, labels):
    plt.plot(s_values, quantity.mean(axis=0), label=label)
    plt.fill_between(
            s_values,
            np.quantile(quantity, 0.1, axis=0),
            np.quantile(quantity, 0.9, axis=0),
            alpha=0.2
        )

plt.xlim([0, 1])
plt.xlabel("Proportion of missing values")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.savefig("../figures/inpainting_patches_psnr.pdf")
