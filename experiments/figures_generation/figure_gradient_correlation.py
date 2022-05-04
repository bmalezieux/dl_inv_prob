import numpy as np
import matplotlib.pyplot as plt


plt.style.use('figures_style_small.mplstyle')


results_final = np.load("../results/gradient_scaling_correlation.npy")
sigma_diag = np.load("../results/gradient_scaling_correlation_sigma.npy")
reg_list = np.load("../results/gradient_scaling_correlation_reg_list.npy")


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
plt.legend()
plt.savefig("../figures/gradient_scaling_correlation.pdf")
