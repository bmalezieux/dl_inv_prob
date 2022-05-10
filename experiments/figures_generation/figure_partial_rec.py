import matplotlib.pyplot as plt
import numpy as np


plt.style.use('figures_style_small.mplstyle')

scores = np.load("../results/scores_partial.npy")
dim_m = np.load("../results/dim_m_partial.npy")
spars = np.load("../results/spars_partial.npy")


colors = ["midnightblue", "indigo", "darkcyan", "darkgreen"]
n = 100

for i in range(scores.shape[1]):
    plt.plot(dim_m, scores[:, i].mean(axis=0), label=spars[i], color=colors[i])
    plt.fill_between(
        dim_m,
        np.quantile(scores[:, i], q=0.1, axis=0),
        np.quantile(scores[:, i], q=0.9, axis=0),
        alpha=0.2,
        color=colors[i]
    )

plt.plot(dim_m, np.sqrt(dim_m / n), label="Perfect", color="black")
plt.xlabel("Dim. measurements")
plt.ylabel("Rec. score")
plt.legend(title="Sparsity", loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig("../figures/score_partial_rec.pdf")
