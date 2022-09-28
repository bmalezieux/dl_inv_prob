# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl


plt.style.use('figures_style_full.mplstyle')
fig, axs = plt.subplots(1, 3)

# Patches
df = pd.read_csv('../results/inpainting_patches.csv')

s_values = pd.unique(df["s_values"])
scores = []
scores_weights = []
psnrs = []
psnrs_corrupted = []
for s_value in s_values:
    df_value = df[df["s_values"] == s_value]
    scores.append(np.array(df_value["scores"]))
    scores_weights.append(np.array(df_value["scores_weights"]))
    psnrs.append(np.array(df_value["psnrs"]))
    psnrs_corrupted.append(np.array(df_value["psnrs_corrupted"]))

scores = np.array(scores)
scores_weights = np.array(scores_weights)
psnrs = np.array(psnrs)
psnrs_corrupted = np.array(psnrs_corrupted)

axs[0].plot(s_values, psnrs.mean(axis=1), label="reconstruction")
axs[0].fill_between(
    s_values,
    np.quantile(psnrs, 0.1, axis=1),
    np.quantile(psnrs, 0.9, axis=1),
    alpha=0.2
)
axs[0].plot(s_values, psnrs_corrupted.mean(axis=1), label="observation")
axs[0].fill_between(
    s_values,
    np.quantile(psnrs_corrupted, 0.1, axis=1),
    np.quantile(psnrs_corrupted, 0.9, axis=1),
    alpha=0.2
)
axs[0].set_yticks([10, 15, 20, 25])
axs[0].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
axs[0].set_xlabel("Prop. of missing pixels")
axs[0].set_ylabel("PSNR (dB)\nInpainting")
axs[0].legend(fontsize=6, loc="upper right")
axs[0].grid()

axs[1].plot(s_values, scores_weights.mean(axis=1))
axs[1].fill_between(
        s_values,
        np.quantile(scores_weights, 0.1, axis=1),
        np.quantile(scores_weights, 0.9, axis=1),
        alpha=0.2
    )
axs[1].set_yticks([0.6, 0.7, 0.8, 0.9])
axs[1].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
axs[1].set_xlabel("Prop. of missing pixels")
axs[1].set_ylabel("Weighted score\nInpainting")
axs[1].grid()


# Compressed sensing
data = pd.read_csv("../results/num_measurements.csv")

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
        scores.append(current_data.iloc[index_score]["score_avg"])
        scores_q1.append(current_data.iloc[index_score]["score_q1"])
        scores_q3.append(current_data.iloc[index_score]["score_q3"])
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

axs[2].legend(title="Dim. m", fontsize=6, title_fontsize=6, loc="lower right")
axs[2].set_ylim([0.6, 1])
axs[2].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
axs[2].set_xticks([1, 2, 3, 4, 5])
axs[2].set_xlabel("Number of matrices")
axs[2].set_ylabel("Rec. score\nCompr. sensing")
axs[2].grid()

plt.tight_layout()
plt.savefig("../figures/figure_3.pdf")
# %%
