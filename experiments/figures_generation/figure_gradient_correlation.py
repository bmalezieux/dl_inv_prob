import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.style.use('figures_style_full.mplstyle')

results = pd.read_pickle("../results/gradient_scaling_correlation.pickle")


results_image = results["results_image"]["results"]
results_synt = results["results_synth"]["results"]
sigma_diag = results["sigma_diag"]["sigma_diag"]
reg_list = results["reg_list"]["reg_list"]

fig, axs = plt.subplots(1, 2)


axs[0].plot(sigma_diag, results_image[:, :, 0].mean(axis=1), label="Unscaled")
axs[0].fill_between(
    sigma_diag,
    np.quantile(results_image[:, :, 0], 0.1, axis=1),
    np.quantile(results_image[:, :, 0], 0.9, axis=1),
    alpha=0.2
)

axs[0].plot(sigma_diag, results_image[:, :, 1].mean(axis=1), label="Scaled")
axs[0].fill_between(
    sigma_diag,
    np.quantile(results_image[:, :, 1], 0.1, axis=1),
    np.quantile(results_image[:, :, 1], 0.9, axis=1),
    alpha=0.2
)

for i in range(len(reg_list)):
    axs[0].plot(sigma_diag, results_image[:, :, i+2].mean(axis=1),
             label=f"Scaled + reg {round(reg_list[i], 2)}")
    axs[0].fill_between(
        sigma_diag,
        np.quantile(results_image[:, :, i+2], 0.1, axis=1),
        np.quantile(results_image[:, :, i+2], 0.9, axis=1),
        alpha=0.2
    )

axs[0].set_xlabel("Variance diag W")
axs[0].set_ylabel("Correlation with g")
axs[0].set_ylim([0.8, 0.9])
axs[0].set_yticks([0.80, 0.82, 0.84, 0.86, 0.88, 0.90])
axs[0].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
axs[0].set_title("Image")
axs[0].grid()


#### Sensing gradient correlation ####

axs[1].plot(sigma_diag, results_synt[:, :, 0].mean(axis=1), label="Unscaled")
axs[1].fill_between(
    sigma_diag,
    np.quantile(results_synt[:, :, 0], 0.1, axis=1),
    np.quantile(results_synt[:, :, 0], 0.9, axis=1),
    alpha=0.2
)

axs[1].plot(sigma_diag, results_synt[:, :, 1].mean(axis=1), label="Scaled")
axs[1].fill_between(
    sigma_diag,
    np.quantile(results_synt[:, :, 1], 0.1, axis=1),
    np.quantile(results_synt[:, :, 1], 0.9, axis=1),
    alpha=0.2
)

for i in range(len(reg_list)):
    axs[1].plot(sigma_diag, results_synt[:, :, i+2].mean(axis=1),
             label=f"Scaled + reg {round(reg_list[i], 2)}")
    axs[1].fill_between(
        sigma_diag,
        np.quantile(results_synt[:, :, i+2], 0.1, axis=1),
        np.quantile(results_synt[:, :, i+2], 0.9, axis=1),
        alpha=0.2
    )

axs[1].set_xlabel("Variance diag cov")
axs[1].set_ylim([0.7, 0.8])
axs[1].set_yticks([0.70, 0.72, 0.74, 0.76, 0.78, 0.80])
axs[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
axs[1].set_title("Sensing")
axs[1].grid()


plt.legend(bbox_to_anchor=(1.05, 1.2), loc='upper left')
plt.savefig("../figures/gradient_scaling_correlation.pdf")
