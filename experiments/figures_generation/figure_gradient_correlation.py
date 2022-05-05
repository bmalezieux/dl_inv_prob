import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.style.use('figures_style_small.mplstyle')

results = pd.read_pickle("../results/gradient_scaling_correlation.pickle")


results_final = results["results"]["results"]
sigma_diag = results["sigma_diag"]["sigma_diag"]
reg_list = results["reg_list"]["reg_list"]


plt.plot(sigma_diag, results_final[:, :, 0].mean(axis=1), label="Unscaled")
plt.fill_between(
    sigma_diag,
    np.quantile(results_final[:, :, 0], 0.1, axis=1),
    np.quantile(results_final[:, :, 0], 0.9, axis=1),
    alpha=0.2
)

plt.plot(sigma_diag, results_final[:, :, 1].mean(axis=1), label="Scaled")
plt.fill_between(
    sigma_diag,
    np.quantile(results_final[:, :, 1], 0.1, axis=1),
    np.quantile(results_final[:, :, 1], 0.9, axis=1),
    alpha=0.2
)

for i in range(len(reg_list)):
    plt.plot(sigma_diag, results_final[:, :, i+2].mean(axis=1),
             label=f"Scaled + reg {round(reg_list[i], 2)}")
    plt.fill_between(
        sigma_diag,
        np.quantile(results_final[:, :, i+2], 0.1, axis=1),
        np.quantile(results_final[:, :, i+2], 0.9, axis=1),
        alpha=0.2
    )

plt.xlabel("Variance diag cov")
plt.ylabel("Correlation with true gradient")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("../figures/gradient_scaling_correlation.pdf")
