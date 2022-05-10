import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('figures_style_small.mplstyle')

df = pd.read_pickle('../results/inpainting_patches.pickle')
scores = df.scores.scores
s_values = df.s_values.s_values


plt.plot(s_values, scores.mean(axis=0))
plt.fill_between(
        s_values,
        np.quantile(scores, 0.1, axis=0),
        np.quantile(scores, 0.9, axis=0),
        alpha=0.2
    )

plt.xlim([0, 1])
plt.ylim([0, 1])
# plt.yticks(np.linspace(0, 1, 6))
plt.xlabel("Proportion of missing values")
plt.ylabel("Rec. score")
plt.savefig("../figures/inpainting_patches_score.pdf")
