import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.style.use('figures_style_small.mplstyle')

scores = np.load("../results/scores_partial.npy")
dim_m = np.load("../results/dim_m_partial.npy")
spars = np.load("../results/spars_partial.npy")

constant_color = 3
c = np.arange(1, len(spars) + constant_color)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])


n = 100

for i in range(scores.shape[1]):
    plt.plot(dim_m, scores[:, i].mean(axis=0),
             label=spars[i], color=cmap.to_rgba(i + constant_color))
    plt.fill_between(
        dim_m,
        np.quantile(scores[:, i], q=0.1, axis=0),
        np.quantile(scores[:, i], q=0.9, axis=0),
        alpha=0.2,
        color=cmap.to_rgba(i+1)
    )

plt.plot(dim_m, np.sqrt(dim_m / n), label="Perfect", color="black")
plt.xlabel("Dim. measurements")
plt.ylabel("Rec. score")
plt.grid()
plt.legend(title="Sparsity", loc="lower right")
plt.savefig("../figures/score_partial_rec.pdf")
