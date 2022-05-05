import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate


plt.style.use('figures_style_small.mplstyle')

results = pd.read_pickle("../results/gradient_scaling.pickle")

times_results = results["times"][:-1]
scores_results = results["scores"][:-1]
loss_results = results["loss"][:-1]
reg_list = results["reg_list"]["reg_list"]

new_times = np.linspace(0, 30, 100)
recoveries = []
for i in range(len(times_results)):
    recoveries.append(np.zeros((len(times_results[i]), len(new_times))))

t_max = np.max(new_times)
for i in range(len(times_results)):
    for j in range(len(times_results[i])):
        if times_results[i][j][-1] < t_max:
            times_results[i][j][-1] = t_max
        f = interpolate.interp1d(times_results[i][j], scores_results[i][j])
        recoveries[i][j] = f(new_times)

recoveries = np.array(recoveries)
max_rec = np.max(recoveries[:, :, -1])

for i in range(recoveries.shape[0]):
    if i == 0:
        plt.plot(
            new_times,
            recoveries[i].mean(axis=0),
            label="Unscaled"
        )
        plt.fill_between(
            new_times,
            np.quantile(recoveries[i], 0.1, axis=0),
            np.quantile(recoveries[i], 0.9, axis=0),
            alpha=0
        )
    elif i == 1:
        plt.plot(
            new_times,
            recoveries[i].mean(axis=0),
            label="Scaled"
        )
        plt.fill_between(
            new_times,
            np.quantile(recoveries[i], 0.1, axis=0),
            np.quantile(recoveries[i], 0.9, axis=0),
            alpha=0
        )
    else:
        plt.plot(
            new_times,
            recoveries[i].mean(axis=0),
            label=f"Scaled + reg {round(reg_list[i-2], 2)}"
        )
        plt.fill_between(
            new_times,
            np.quantile(recoveries[i], 0.1, axis=0),
            np.quantile(recoveries[i], 0.9, axis=0),
            alpha=0
        )
plt.xlabel("Time (s)")
plt.ylabel("Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("../figures/gradient_scaling.pdf")
