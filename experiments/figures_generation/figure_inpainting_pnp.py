# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('figures_style_full.mplstyle')

data = pd.read_csv("../results/inpainting_pnp.csv")

# %%

fig, axs = plt.subplots(1, 4)

# PSNR
for i, sigma in enumerate(pd.unique(data["sigma_sample"])):
    psnr_unsupervised = []
    psnr_supervised = []
    psnr_corr = []
    for rho in list(pd.unique(data["prop"]))[:-1]:
        current_psnr = data[
            (data["prop"] == rho)
            & (data["sigma_sample"] == sigma)
        ]["psnr_rec_unsupervised"].max()
        psnr_unsupervised.append(current_psnr)

        current_psnr = data[
            (data["prop"] == rho)
            & (data["sigma_sample"] == sigma)
        ]["psnr_rec_supervised"].max()
        psnr_supervised.append(current_psnr)

        current_psnr = data[
            (data["prop"] == rho)
            & (data["sigma_sample"] == sigma)
        ]["psnr_corr"].max()
        psnr_corr.append(current_psnr)

    axs[i].plot(list(pd.unique(data["prop"]))[:-1], psnr_unsupervised, label="Unsupervised")
    axs[i].plot(list(pd.unique(data["prop"]))[:-1], psnr_supervised, label="Supervised")
    axs[i].plot(list(pd.unique(data["prop"]))[:-1], psnr_corr, label="Corrupted")


    axs[i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma ** 2)) / np.log(10), 0)}")
    axs[i].grid()
    if i == 0:
        axs[i].set_ylabel("PSNR")
    axs[i].set_xlabel("Prop. of missing pixels")


legend = axs[0].legend()
handles, labels = axs[0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.23, 0.56))
plt.tight_layout()
plt.savefig("../figures/inpainting_pnp_full.pdf")
# %%
