# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('figures_style_full.mplstyle')

data = pd.read_csv("../results/deblurring_pnp_woof.csv")

# %%

fig, axs = plt.subplots(1, 4)

# PSNR
for i, sigma in enumerate(pd.unique(data["sigma_sample"])):
    psnr_unsupervised = []
    psnr_supervised = []
    for sigma_blurr in list(pd.unique(data["sigma_blurr"])):
        current_psnr = data[
            (data["sigma_blurr"] == sigma_blurr)
            & (data["sigma_sample"] == sigma)
        ]["psnr_rec_unsupervised"].max()
        psnr_unsupervised.append(current_psnr)

        current_psnr = data[
            (data["sigma_blurr"] == sigma_blurr)
            & (data["sigma_sample"] == sigma)
        ]["psnr_rec_supervised"].max()
        psnr_supervised.append(current_psnr)


    axs[i].plot(list(pd.unique(data["sigma_blurr"])), psnr_unsupervised, label="Unsupervised PnP")
    axs[i].plot(list(pd.unique(data["sigma_blurr"])), psnr_supervised, label="Supervised PnP")


    axs[i].set_title(f"SNR {round(10 * np.log(0.205 ** 2 / (sigma ** 2)) / np.log(10), 0)}")
    axs[i].grid()
    if i == 0:
        axs[i].set_ylabel("PSNR")
    axs[i].set_xlabel("Sigma blurr")


legend = axs[0].legend()
handles, labels = axs[0].get_legend_handles_labels()
legend.remove()
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.25, 0.56))
plt.tight_layout()
plt.show()
plt.savefig("../figures/deblurring_pnp_full.pdf")
# %%

fig, axs = plt.subplots(1, 3)

sigma = 0.02

psnr_unsupervised = []
psnr_supervised = []

psnr_ker_unsupervised = []
psnr_ker_supervised = []

psnr_ran_unsupervised = []
psnr_ran_supervised = []

for sigma_blurr in list(pd.unique(data["sigma_blurr"])):
    current_data = data[
        (data["sigma_blurr"] == sigma_blurr)
        & (data["sigma_sample"] == sigma)
    ]
    index = current_data["psnr_rec_unsupervised"].argmax()
    psnr_unsupervised.append(current_data.iloc[index]["psnr_rec_unsupervised"])
    psnr_ker_unsupervised.append(current_data.iloc[index]["psnr_ker_unsupervised"])
    psnr_ran_unsupervised.append(current_data.iloc[index]["psnr_ran_unsupervised"])

    current_data = data[
        (data["sigma_blurr"] == sigma_blurr)
        & (data["sigma_sample"] == sigma)
    ]
    index = current_data["psnr_rec_supervised"].argmax()
    psnr_supervised.append(current_data.iloc[index]["psnr_rec_supervised"])
    psnr_ker_supervised.append(current_data.iloc[index]["psnr_ker_supervised"])
    psnr_ran_supervised.append(current_data.iloc[index]["psnr_ran_supervised"])


axs[0].plot(list(pd.unique(data["sigma_blurr"])), psnr_unsupervised, label="Unsupervised PnP")
axs[0].plot(list(pd.unique(data["sigma_blurr"])), psnr_supervised, label="Supervised PnP")
axs[0].grid()
axs[0].set_ylabel("SNR (dB)")
axs[0].set_xlabel("Sigma blurr")
axs[0].set_title("SNR full space")

axs[1].plot(list(pd.unique(data["sigma_blurr"])), 10 + np.array(psnr_ker_unsupervised), label="Unsupervised PnP")
axs[1].plot(list(pd.unique(data["sigma_blurr"])), 10 + np.array(psnr_ker_supervised), label="Supervised PnP")
axs[1].grid()
axs[1].set_xlabel("Sigma blurr")
axs[1].set_title("SNR kernel space")

axs[2].plot(list(pd.unique(data["sigma_blurr"])), psnr_ran_unsupervised, label="Unsupervised PnP")
axs[2].plot(list(pd.unique(data["sigma_blurr"])), psnr_ran_supervised, label="Supervised PnP")
axs[2].grid()
axs[2].set_xlabel("Sigma blurr")
axs[2].set_title("SNR range space")

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
fig.legend(labels=labels, handles=handles, loc="center right", bbox_to_anchor=(1.25, 0.56))
plt.tight_layout()
plt.savefig("../figures/deblurring_pnp_spaces.pdf")

# %%
