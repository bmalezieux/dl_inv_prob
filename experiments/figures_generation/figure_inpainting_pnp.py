# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('figures_style_full.mplstyle')

data = pd.read_csv("../results/inpainting_pnp_samples_woof.csv")

# %%

fig, axs = plt.subplots(3, 4, figsize=(6.5, 4.5))

# PSNR
for i, sigma in enumerate(pd.unique(data["sigma_sample"])):
    psnr_unsupervised = []
    psnr_supervised = []

    psnr_ker_unsupervised = []
    psnr_ker_supervised = []

    psnr_ran_unsupervised = []
    psnr_ran_supervised = []
    for rho in list(pd.unique(data["prop"])):
        current_data = data[
            (data["prop"] == rho)
            & (data["sigma_sample"] == sigma)
        ]
        index = current_data["psnr_rec_unsupervised"].argmax()

        psnr_unsupervised.append(current_data.iloc[index]["psnr_rec_unsupervised"])
        psnr_ker_unsupervised.append(current_data.iloc[index]["psnr_ker_unsupervised"])
        psnr_ran_unsupervised.append(current_data.iloc[index]["psnr_ran_unsupervised"])

        index = current_data["psnr_rec_supervised"].argmax()
        psnr_supervised.append(current_data.iloc[index]["psnr_rec_supervised"])
        psnr_ker_supervised.append(current_data.iloc[index]["psnr_ker_supervised"])
        psnr_ran_supervised.append(current_data.iloc[index]["psnr_ran_supervised"])


    axs[0, i].plot(pd.unique(data["prop"]), psnr_supervised, label="Sup. PnP", linestyle="dashed")
    axs[0, i].plot(pd.unique(data["prop"]), psnr_unsupervised, label="Unsup. PnP", linestyle="dashed")
    axs[0, i].grid()

    axs[1, i].plot(pd.unique(data["prop"]), psnr_ker_supervised, label="Supervised PnP", linestyle="dashed")
    axs[1, i].plot(pd.unique(data["prop"]), psnr_ker_unsupervised, label="Unsupervised PnP", linestyle="dashed")
    axs[1, i].grid()

    axs[2, i].plot(pd.unique(data["prop"]), psnr_ran_supervised, label="Supervised PnP", linestyle="dashed")
    axs[2, i].plot(pd.unique(data["prop"]), psnr_ran_unsupervised, label="Unsupervised PnP", linestyle="dashed")
    axs[2, i].grid()

    axs[0, i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma ** 2)) / np.log(10), 0)}")

    if i == 0:
        axs[0, i].set_ylabel("PSNR recovery")
        axs[1, i].set_ylabel("PSNR kernel")
        axs[2, i].set_ylabel("PSNR range")

    axs[2, i].set_xlabel("Prop. of missing pixels")


legend = axs[0, 0].legend()
handles, labels = axs[0, 0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.20, 0.52))
plt.tight_layout()
plt.savefig("../figures/inpainting_pnp_full_2.pdf")
plt.show()
plt.clf()
# %%

fig, axs = plt.subplots(1, 3)

sigma = 0.02

psnr_unsupervised = []
psnr_supervised = []

psnr_ker_unsupervised = []
psnr_ker_supervised = []

psnr_ran_unsupervised = []
psnr_ran_supervised = []

for rho in list(pd.unique(data["prop"])):
    current_data = data[
        (data["prop"] == rho)
        & (data["sigma_sample"] == sigma)
    ]
    index = current_data["psnr_rec_unsupervised"].argmax()
    psnr_unsupervised.append(current_data.iloc[index]["psnr_rec_unsupervised"])
    psnr_ker_unsupervised.append(current_data.iloc[index]["psnr_ker_unsupervised"])
    psnr_ran_unsupervised.append(current_data.iloc[index]["psnr_ran_unsupervised"])

    current_data = data[
        (data["prop"] == rho)
        & (data["sigma_sample"] == sigma)
    ]
    index = current_data["psnr_rec_supervised"].argmax()
    psnr_supervised.append(current_data.iloc[index]["psnr_rec_supervised"])
    psnr_ker_supervised.append(current_data.iloc[index]["psnr_ker_supervised"])
    psnr_ran_supervised.append(current_data.iloc[index]["psnr_ran_supervised"])


axs[0].plot(list(pd.unique(data["prop"])), psnr_supervised, label="Supervised PnP", linestyle="dashed")
axs[0].plot(list(pd.unique(data["prop"])), psnr_unsupervised, label="Unsupervised PnP", linestyle="dashed")

axs[0].grid()
axs[0].set_ylabel("PSNR (dB)")
axs[0].set_xlabel("Prop. of missing pixels")
# axs[0].set_title("PSNR full space")

axs[1].plot(list(pd.unique(data["prop"])), psnr_ker_supervised, label="Supervised PnP", linestyle="dashed")
axs[1].plot(list(pd.unique(data["prop"])), psnr_ker_unsupervised, label="Unsupervised PnP", linestyle="dashed")

axs[1].grid()
axs[1].set_xlabel("Prop. of missing pixels")
# axs[1].set_title("PSNR kernel space")

axs[2].plot(list(pd.unique(data["prop"])), psnr_ran_supervised, label="Supervised PnP", linestyle="dashed")
axs[2].plot(list(pd.unique(data["prop"])), psnr_ran_unsupervised, label="Unsupervised PnP", linestyle="dashed")

axs[2].grid()
axs[2].set_xlabel("Prop. of missing pixels")
# axs[2].set_title("PSNR range space")

axs[0].set_ylim([10, 40])
axs[0].set_yticks([15, 20, 25, 30, 35])
axs[0].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

axs[1].set_ylim([10, 40])
axs[1].set_yticks([15, 20, 25, 30, 35])
axs[1].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

axs[2].set_ylim([10, 40])
axs[2].set_yticks([15, 20, 25, 30, 35])
axs[2].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])


legend = axs[0].legend()
handles, labels = axs[0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.25, 0.56))
plt.tight_layout()
plt.savefig("../figures/inpainting_pnp_spaces.pdf")

# %%
