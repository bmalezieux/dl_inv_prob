import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('figures_style_full.mplstyle')

df = pd.read_pickle('../results/inpainting_patches.pickle')
psnrs = df.psnrs.psnrs
scores = df.scores.scores
psnrs_corrupted = df.psnrs_corrupted.psnrs_corrupted
s_values = df.s_values.s_values
quantities = [psnrs, psnrs_corrupted]
labels = ['reconstruction', 'corrupted image']

fig, (ax1, ax2) = plt.subplots(1, 2)

for quantity, label in zip(quantities, labels):
    ax1.plot(s_values, quantity.mean(axis=0), label=label)
    ax1.fill_between(
            s_values,
            np.quantile(quantity, 0.1, axis=0),
            np.quantile(quantity, 0.9, axis=0),
            alpha=0.2
        )
ax1.set_xlim([0, 1])
ax1.set_xlabel("Proportion of missing values")
ax1.set_ylabel("PSNR (dB)")
ax1.legend(loc='upper right', fontsize=5)
ax1.grid()

ax2.plot(s_values, scores.mean(axis=0))
ax2.fill_between(
        s_values,
        np.quantile(scores, 0.1, axis=0),
        np.quantile(scores, 0.9, axis=0),
        alpha=0.2
    )

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_xlabel("Proportion of missing values")
ax2.set_ylabel("Rec. score")
ax2.grid()

plt.savefig("../figures/inpainting_patches.pdf")
