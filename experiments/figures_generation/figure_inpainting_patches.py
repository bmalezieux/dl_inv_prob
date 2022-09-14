# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('figures_style_full.mplstyle')

df = pd.read_csv('../results/inpainting_patches.csv')

fig, axs = plt.subplots(1, 2)

s_values = pd.unique(df["s_values"])
scores = []
scores_weights = []
psnrs = []
psnrs_corrupted = []
for s_value in s_values:
    df_value = df[df["s_values"] == s_value]
    scores.append(np.array(df_value["scores"]))
    scores_weights.append(np.array(df_value["scores_weights"]))
    psnrs.append(np.array(df_value["psnrs"]))
    psnrs_corrupted.append(np.array(df_value["psnrs_corrupted"]))

scores = np.array(scores)
scores_weights = np.array(scores_weights)
psnrs = np.array(psnrs)
psnrs_corrupted = np.array(psnrs_corrupted)


axs[0].plot(s_values, psnrs.mean(axis=1), label="reconstruction")
axs[0].fill_between(
    s_values,
    np.quantile(psnrs, 0.1, axis=1),
    np.quantile(psnrs, 0.9, axis=1),
    alpha=0.2
)
axs[0].plot(s_values, psnrs_corrupted.mean(axis=1), label="corrupted image")
axs[0].fill_between(
    s_values,
    np.quantile(psnrs_corrupted, 0.1, axis=1),
    np.quantile(psnrs_corrupted, 0.9, axis=1),
    alpha=0.2
)
axs[0].set_yticks([10, 15, 20, 25])
axs[0].set_xlabel("Prop. of missing pixels")
axs[0].set_ylabel("PSNR (dB)")
axs[0].legend(fontsize=6)
axs[0].grid()

axs[1].plot(s_values, scores_weights.mean(axis=1))
axs[1].fill_between(
        s_values,
        np.quantile(scores_weights, 0.1, axis=1),
        np.quantile(scores_weights, 0.9, axis=1),
        alpha=0.2
    )
axs[1].set_yticks([0.6, 0.7, 0.8, 0.9])
axs[1].set_xlabel("Prop. of missing pixels")
axs[1].set_ylabel("Score with weights")
axs[1].grid()

plt.savefig("../figures/inpainting_patches.pdf")

# %%