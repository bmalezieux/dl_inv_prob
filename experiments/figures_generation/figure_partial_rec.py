# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('figures_style_small.mplstyle')
data = pd.read_csv("../results/partial_rec.csv")

# %%

n = 100

constant_color = 3
c = np.arange(1, len(pd.unique(data["sparsity"])) + constant_color)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

for i, spars in enumerate(pd.unique(data["sparsity"])):
    scores = []
    scores_q1 = []
    scores_q3 = []
    for m in pd.unique(data["dim_measurement"]):
        current_data = data[
            (data["dim_measurement"] == m)
            & (data["sparsity"] == spars)
        ]
        index_score = current_data["score_avg"].argmax()
        scores.append(current_data.iloc[index_score]["score_avg"])
        scores_q1.append(current_data.iloc[index_score]["score_q1"])
        scores_q3.append(current_data.iloc[index_score]["score_q3"])
    plt.plot(
        pd.unique(data["dim_measurement"]),
        scores,
        label=spars,
        color=cmap.to_rgba(i + constant_color)
    )
    plt.fill_between(
        pd.unique(data["dim_measurement"]),
        scores_q1,
        scores_q3,
        alpha=0.2,
        color=cmap.to_rgba(i+constant_color)
    )

plt.plot(pd.unique(data["dim_measurement"]), np.sqrt(pd.unique(data["dim_measurement"]) / n), label="Perfect", color="black")
plt.legend(title="Sparsity", loc="lower right")
plt.xlabel("Dim. m")
plt.ylabel("Rec. score")
plt.grid()
plt.savefig("../figures/score_partial_rec.pdf")


# %%
