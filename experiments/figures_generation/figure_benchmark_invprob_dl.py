# %%
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd

# %%
results = pd.read_csv("../results/results_inpainting.csv")
results.head()

values_p = results["p"].unique()
values_sparsity = results["sparsity"].unique()
values_n_samples = results["n_samples"].unique()

rec_score = 0.95

plt.style.use('figures_style_full.mplstyle')

fig, axs = plt.subplots(1, 3)

xx, yy = np.meshgrid(values_p, values_sparsity)
samples = values_n_samples.max() * np.ones((len(values_p), len(values_sparsity)))

for i, p in enumerate(values_p):
    for j, s in enumerate(values_sparsity):
        for n in values_n_samples:
            score = results.loc[(results["p"] == p) & (results["n_samples"] == n) & (results["sparsity"] == s)]["score_avg"].max()
            if score > rec_score:
                samples[i, j] = n
                break

im0 = axs[0].contourf(xx, yy, samples.T, locator=ticker.LogLocator(subs="all"), cmap="RdBu_r")
axs[0].set_xlabel("p")
axs[0].set_ylabel("sparsity")
axs[0].set_title("N samples for recovery")
cb = fig.colorbar(im0, ax=axs[0])
cb.set_ticks([1e2, 1e3, 1e4])

xx, yy = np.meshgrid(values_p, values_n_samples)
sparsities = np.zeros((len(values_p), len(values_n_samples)))

for i, p in enumerate(values_p):
    for j, n in enumerate(values_n_samples):
        for s in values_sparsity[::-1]:
            score = results.loc[(results["p"] == p) & (results["n_samples"] == n) & (results["sparsity"] == s)]["score_avg"].max()
            if score > rec_score:
                sparsities[i, j] = s
                break

im1 = axs[1].contourf(xx, yy, sparsities.T, cmap="RdBu_r")
axs[1].set_xlabel("p")
axs[1].set_ylabel("n samples")
axs[1].set_yscale("log")
axs[1].set_title("Max. sparsity for recovery")
cb = fig.colorbar(im1, ax=axs[1])
cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])


xx, yy = np.meshgrid(values_sparsity, values_n_samples)
proportions = np.ones((len(values_sparsity), len(values_n_samples)))

for i, s in enumerate(values_sparsity):
    for j, n in enumerate(values_n_samples):
        for p in values_p:
            score = results.loc[(results["p"] == p) & (results["n_samples"] == n) & (results["sparsity"] == s)]["score_avg"].max()
            if score > rec_score:
                proportions[i, j] = p
                break

im2 = axs[2].contourf(xx, yy, proportions.T, cmap="RdBu_r")
axs[2].set_xlabel("sparsity")
axs[2].set_ylabel("n samples")
axs[2].set_yscale("log")
axs[2].set_title("Min. proportion for recovery")
cb = fig.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.savefig("../figures/inpainting_heatmaps.pdf")
    

# %%
