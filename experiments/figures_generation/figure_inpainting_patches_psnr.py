import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('figures_style_small.mplstyle')

df = pd.read_pickle('../results/inpainting_patches.pickle')
psnrs = df.psnrs.psnrs
s_values = df.s_values.s_values


plt.plot(s_values, psnrs.mean(axis=0))
plt.fill_between(
        s_values,
        np.quantile(psnrs, 0.1, axis=0),
        np.quantile(psnrs, 0.9, axis=0),
        alpha=0.2
    )

plt.xlim([0, 1])
# plt.yticks(np.linspace(0, 1, 6))
plt.xlabel("Proportion of missing values")
plt.ylabel("PSNR (dB)")
plt.savefig("../figures/inpainting_patches_psnr.pdf")
