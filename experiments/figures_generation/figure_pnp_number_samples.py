# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('figures_style_full.mplstyle')

data = pd.read_csv("../results/inpainting_pnp_number_samples_woof.csv")

# %%

# PSNR

fig, axs = plt.subplots(1, len(pd.unique(data["sigma_sample"])))

for i, sigma_sample in enumerate(list(pd.unique(data["sigma_sample"]))):
    psnr_unsupervised = []
    psnr_supervised = []
    for n_sample in list(pd.unique(data["n_samples"])):
        current_psnr = data[
            (data["n_samples"] == n_sample)
            & (data["sigma_sample"] == sigma_sample)
        ]["psnr_rec_unsupervised"].max()
        psnr_unsupervised.append(current_psnr)

        current_psnr = data[
            (data["n_samples"] == n_sample)
            & (data["sigma_sample"] == sigma_sample)
        ]["psnr_rec_supervised"].max()
        psnr_supervised.append(current_psnr)
    axs[i].plot(list(pd.unique(data["n_samples"])), psnr_unsupervised, label="Unsupervised")
    axs[i].plot(list(pd.unique(data["n_samples"])), psnr_supervised, label="Supervised")
    axs[i].set_xlabel("N. samples")
    axs[i].set_ylabel("PSNR")
    axs[i].set_xlim([10, 1000])
    axs[i].set_xscale("log")
    axs[i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma_sample ** 2)) / np.log(10), 0)}")
    axs[i].grid()

legend = axs[0].legend()
handles, labels = axs[0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.23, 0.56))
plt.tight_layout()
plt.savefig("../figures/inpainting_pnp_samples.png")
# %%
