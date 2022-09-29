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
fig, axs = plt.subplots(3, 4, figsize=(6, 4.5))

# PSNR
for i, sigma in enumerate(pd.unique(data_256["sigma"])):
    for name in pd.unique(data_supervised_256["name"]):
        psnr = []
        psnr_ker = []
        psnr_ran = []
        for sigma_blurr in list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1]:
            current_data = data_supervised_256[
                (data_supervised_256["sigma_blurr"] == sigma_blurr)
                & (data_supervised_256["sigma"] == sigma)
                & (data_supervised_256["name"] == name)
            ]
            id = current_data["psnr_rec"].argmax()
            psnr.append(current_data.iloc[id]["psnr_rec"])
            psnr_ker.append(current_data.iloc[id]["psnr_ker"])
            psnr_ran.append(current_data.iloc[id]["psnr_ran"])
        axs[0, i].plot(list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1], psnr, label=name)
        axs[1, i].plot(list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1], 10 + np.array(psnr_ker), label=name)
        axs[2, i].plot(list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1], psnr_ran, label=name)
    for name in pd.unique(data_256["name"]):
        psnr = []
        psnr_ker = []
        psnr_ran = []
        for sigma_blurr in list(pd.unique(data_256["sigma_blurr"]))[:-1]:
            current_data = data_256[
                (data_256["sigma_blurr"] == sigma_blurr)
                & (data_256["sigma"] == sigma)
                & (data_256["name"] == name)
            ]
            id = current_data["psnr_rec"].argmax()
            psnr.append(current_data.iloc[id]["psnr_rec"])
            psnr_ker.append(current_data.iloc[id]["psnr_ker"])
            psnr_ran.append(current_data.iloc[id]["psnr_ran"])
        axs[0, i].plot(list(pd.unique(data_256["sigma_blurr"]))[:-1], psnr, label=name)
        axs[1, i].plot(list(pd.unique(data_256["sigma_blurr"]))[:-1], 10 + np.array(psnr_ker), label=name)
        axs[2, i].plot(list(pd.unique(data_256["sigma_blurr"]))[:-1], psnr_ran, label=name)

    axs[0, i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma ** 2)) / np.log(10), 0)}")
    axs[0, i].grid()
    axs[1, i].grid()
    axs[2, i].grid()
    if i == 0:
        axs[0, i].set_ylabel("PSNR recovery")
        axs[1, i].set_ylabel("PSNR kernel")
        axs[2, i].set_ylabel("PSNR range")
    axs[2, i].set_xlabel("Sigma blurr")

# # IS divergence
# for i, sigma in enumerate(pd.unique(data_256["sigma"])):
#     for name in pd.unique(data_supervised_256["name"]):
#         is_div = []
#         for sigma_blurr in list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1]:
#             current_is = data_supervised_256[
#                 (data_supervised_256["sigma_blurr"] == sigma_blurr)
#                 & (data_supervised_256["sigma"] == sigma)
#                 & (data_supervised_256["name"] == name)
#             ]["is_rec"].min()
#             is_div.append(current_is)
#         axs[1, i].plot(list(pd.unique(data_supervised_256["sigma_blurr"]))[:-1], is_div, label=name)
#     for name in pd.unique(data_256["name"]):
#         is_div = []
#         for sigma_blurr in list(pd.unique(data_256["sigma_blurr"]))[:-1]:
#             current_is = data_256[
#                 (data_256["sigma_blurr"] == sigma_blurr)
#                 & (data_256["sigma"] == sigma)
#                 & (data_256["name"] == name)
#             ]["is_rec"].min()
#             is_div.append(current_is)
#         axs[1, i].plot(list(pd.unique(data_256["sigma_blurr"]))[:-1], is_div, label=name)

#     axs[1, i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma ** 2)) / np.log(10), 0)}")
#     axs[1, i].grid()
#     if i == 0:
#         axs[1, i].set_ylabel("IS div.")
#     axs[1, i].set_xlabel("Sigma blurr")

legend = axs[0, 0].legend()
handles, labels = axs[0, 0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.23, 0.52))
plt.tight_layout()
plt.savefig("../figures/deblurring_single_image_full_2.pdf")
plt.clf()
# %%

fig, axs = plt.subplots(1, 3)

sigma = 0.02
for name in pd.unique(data_supervised_256["name"]):
    psnr = []
    psnr_ker = []
    psnr_ran = []
    for sigma_blurr in list(pd.unique(data_supervised_256["sigma_blurr"])):
        current_data = data_supervised_256[
            (data_supervised_256["sigma_blurr"] == sigma_blurr)
            & (data_supervised_256["sigma"] == sigma)
            & (data_supervised_256["name"] == name)
        ]
        id = current_data["psnr_rec"].argmax()
        psnr.append(current_data.iloc[id]["psnr_rec"])
        psnr_ker.append(current_data.iloc[id]["psnr_ker"])
        psnr_ran.append(current_data.iloc[id]["psnr_ran"])
    axs[0].plot(np.array(list(pd.unique(data_supervised_256["sigma_blurr"]))), psnr, label=name)
    axs[1].plot(np.array(list(pd.unique(data_supervised_256["sigma_blurr"]))), 10 + np.array(psnr_ker), label=name)
    axs[2].plot(np.array(list(pd.unique(data_supervised_256["sigma_blurr"]))), psnr_ran, label=name)
for name in pd.unique(data_256["name"]):
    psnr = []
    psnr_ker = []
    psnr_ran = []
    for sigma_blurr in list(pd.unique(data_256["sigma_blurr"])):
        current_data = data_256[
            (data_256["sigma_blurr"] == sigma_blurr)
            & (data_256["sigma"] == sigma)
            & (data_256["name"] == name)
        ]
        id = current_data["psnr_rec"].argmax()
        psnr.append(current_data.iloc[id]["psnr_rec"])
        psnr_ker.append(current_data.iloc[id]["psnr_ker"])
        psnr_ran.append(current_data.iloc[id]["psnr_ran"])
    axs[0].plot(np.array(list(pd.unique(data_256["sigma_blurr"]))), psnr, label=name)
    axs[1].plot(np.array(list(pd.unique(data_256["sigma_blurr"]))), 10 + np.array(psnr_ker), label=name)
    axs[2].plot(np.array(list(pd.unique(data_256["sigma_blurr"]))), psnr_ran, label=name)

axs[0].set_title("SNR full space")
axs[1].set_title("SNR kernel space")
axs[2].set_title("SNR range space")
axs[0].grid()
axs[1].grid()
axs[2].grid()

axs[0].set_ylabel("SNR (dB)")
# axs[0].set_xlabel("Sigma blurr")
# axs[1].set_xlabel("Sigma blurr")
# axs[2].set_xlabel("Sigma blurr")

axs[0].set_ylim([5, 40])
axs[0].set_yticks([10, 15, 20, 25, 30, 35])
axs[0].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

axs[1].set_ylim([5, 40])
axs[1].set_yticks([10, 15, 20, 25, 30, 35])
axs[1].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

axs[2].set_ylim([5, 40])
axs[2].set_yticks([10, 15, 20, 25, 30, 35])
axs[2].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

legend = axs[0].legend()
handles, labels = axs[0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.23, 0.56))
plt.tight_layout()
plt.savefig("../figures/deblurring_single_image_spaces.pdf")

# %%
