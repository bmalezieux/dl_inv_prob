import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('figures_style_small.mplstyle')

scores = np.load("../results/compressed_sensing_n_matrices.npy")
m_values = np.load("../results/compressed_sensing_m_values.npy")
N_mat_values = np.load("../results/compressed_sensing_N_mat_values.npy")

constant_color = 4
c = np.arange(1, len(N_mat_values) + constant_color)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

for i in range(scores.shape[1]):
    plt.plot(N_mat_values, scores[:, i, :].mean(axis=0),
             label=m_values[i], color=cmap.to_rgba(i + constant_color))
    plt.fill_between(
        N_mat_values,
        np.quantile(scores[:, i, :], 0.1, axis=0),
        np.quantile(scores[:, i, :], 0.9, axis=0),
        alpha=0.2,
        color=cmap.to_rgba(i+constant_color)
    )
plt.legend(title="Dim. m")
plt.ylim([0.5, 1])
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.xlabel("Number of matrices")
plt.ylabel("Rec. score")
plt.grid()
plt.savefig("../figures/number_measurements_compressed_sensing.pdf")
