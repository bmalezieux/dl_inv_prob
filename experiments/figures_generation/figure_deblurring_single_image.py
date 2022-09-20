# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('figures_style_full.mplstyle')

data = pd.read_csv("../results/deblurring_single_image.csv")
data_256 = data[data["size"] == 256]

# %%

data_supervised = pd.read_csv("../results/deblurring_single_image_supervised.csv")
data_supervised_256 = data_supervised[data_supervised["size"] == 256]

# %%
fig, axs = plt.subplots(2, 4, figsize=(5.5, 3))

# PSNR
for i, sigma in enumerate(pd.unique(data_256["sigma"])):
    for name in pd.unique(data_supervised_256["name"]):
        psnr = []
        for sigma_blurr in list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1]:
            current_psnr = data_supervised_256[
                (data_supervised_256["sigma_blurr"] == sigma_blurr)
                & (data_supervised_256["sigma"] == sigma)
                & (data_supervised_256["name"] == name)
            ]["psnr_rec"].max()
            psnr.append(current_psnr)
        axs[0, i].plot(list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1], psnr, label=name)
    for name in pd.unique(data_256["name"]):
        psnr = []
        for sigma_blurr in list(pd.unique(data_256["sigma_blurr"]))[:-1]:
            current_psnr = data_256[
                (data_256["sigma_blurr"] == sigma_blurr)
                & (data_256["sigma"] == sigma)
                & (data_256["name"] == name)
            ]["psnr_rec"].max()
            psnr.append(current_psnr)
        axs[0, i].plot(list(pd.unique(data_256["sigma_blurr"]))[:-1], psnr, label=name)

    axs[0, i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma ** 2)) / np.log(10), 0)}")
    axs[0, i].grid()
    if i == 0:
        axs[0, i].set_ylabel("PSNR")
    axs[0, i].set_xlabel("Sigma blurr")

# IS divergence
for i, sigma in enumerate(pd.unique(data_256["sigma"])):
    for name in pd.unique(data_supervised_256["name"]):
        is_div = []
        for sigma_blurr in list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1]:
            current_is = data_supervised_256[
                (data_supervised_256["sigma_blurr"] == sigma_blurr)
                & (data_supervised_256["sigma"] == sigma)
                & (data_supervised_256["name"] == name)
            ]["is_rec"].min()
            is_div.append(current_is)
        axs[1, i].plot(list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1], is_div, label=name)
    for name in pd.unique(data_256["name"]):
        is_div = []
        for sigma_blurr in list(pd.unique(data_256["sigma_blurr"]))[:-1]:
            current_is = data_256[
                (data_256["sigma_blurr"] == sigma_blurr)
                & (data_256["sigma"] == sigma)
                & (data_256["name"] == name)
            ]["is_rec"].min()
            is_div.append(current_is)
        axs[1, i].plot(list(pd.unique(data_256["sigma_blurr"]))[:-1], is_div, label=name)

    axs[1, i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma ** 2)) / np.log(10), 0)}")
    axs[1, i].grid()
    if i == 0:
        axs[1, i].set_ylabel("IS div.")
    axs[1, i].set_xlabel("Sigma blurr")

legend = axs[0, 0].legend()
handles, labels = axs[0, 0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.23, 0.56))
plt.tight_layout()
plt.savefig("../figures/deblurring_single_image_full.pdf")
plt.clf()
# %%

fig, axs = plt.subplots(1, 2)

# PSNR
for i, sigma in enumerate(pd.unique(data_256["sigma"])[[1, 3]]):
    for name in pd.unique(data_supervised_256["name"]):
        psnr = []
        for sigma_blurr in list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1]:
            current_psnr = data_supervised_256[
                (data_supervised_256["sigma_blurr"] == sigma_blurr)
                & (data_supervised_256["sigma"] == sigma)
                & (data_supervised_256["name"] == name)
            ]["psnr_rec"].max()
            psnr.append(current_psnr)
        axs[i].plot(np.array(list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1]), psnr, label=name)
    for name in pd.unique(data_256["name"]):
        psnr = []
        for sigma_blurr in list(pd.unique(data_256["sigma_blurr"]))[:-1]:
            current_psnr = data_256[
                (data_256["sigma_blurr"] == sigma_blurr)
                & (data_256["sigma"] == sigma)
                & (data_256["name"] == name)
            ]["psnr_rec"].max()
            psnr.append(current_psnr)
        axs[i].plot(np.array(list(pd.unique(data_256["sigma_blurr"]))[:-1]), psnr, label=name)

    axs[i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma ** 2)) / np.log(10), 0)}")
    axs[i].grid()
    if i == 0:
        axs[i].set_ylabel("PSNR (dB)")
    axs[i].set_xlabel("Sigma blurr")
    if i == 0:
        axs[i].set_ylim([20, 30])
        axs[i].set_yticks([20, 22, 24, 26, 28, 30])
    else:
        axs[i].set_ylim([10, 25])
        axs[i].set_yticks([10, 15, 20, 25])

legend = axs[0].legend()
handles, labels = axs[0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.23, 0.56))
plt.tight_layout()
plt.savefig("../figures/deblurring_single_image.pdf")

# %%