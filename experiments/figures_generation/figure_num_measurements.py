# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.style.use('figures_style_small.mplstyle')
data = pd.read_csv("../results/num_measurements.csv")

# %%
constant_color = 4
c = np.arange(1, len(pd.unique(data["dim_measurement"])) + constant_color)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

for i, m in enumerate(pd.unique(data["dim_measurement"])):
    scores = []
    scores_q1 = []
    scores_q3 = []
    for n_matrice in pd.unique(data["n_matrices"]):
        current_data = data[
            (data["dim_measurement"] == m)
            & (data["n_matrices"] == n_matrice)
        ]
        index_score = current_data["score_avg"].argmax()
        scores.append(1 - current_data.iloc[index_score]["score_avg"])
        scores_q1.append(1 - current_data.iloc[index_score]["score_q1"])
        scores_q3.append(1 - current_data.iloc[index_score]["score_q3"])
    plt.plot(
        pd.unique(data["n_matrices"]),
        scores,
        label=m,
        color=cmap.to_rgba(i + constant_color)
    )
    plt.fill_between(
        pd.unique(data["n_matrices"]),
        scores_q1,
        scores_q3,
        alpha=0.2,
        color=cmap.to_rgba(i+constant_color)
    )

plt.legend(title="Dim. m")
# plt.ylim([0.4, 1])
# plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
plt.yscale("log")
plt.xlabel("Number of matrices")
plt.ylabel("1 - Rec. score")
plt.grid()
# plt.show()
plt.savefig("../figures/number_measurements_compressed_sensing_full.pdf")

# %%
