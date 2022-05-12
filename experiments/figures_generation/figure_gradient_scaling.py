import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate


plt.style.use('figures_style_full.mplstyle')

results = pd.read_pickle("../results/gradient_scaling.pickle")


times_results_synt = list(results["results_synt"]["results"]["times"].values())
scores_results_synt = list(results["results_synt"]["results"]["scores"].values())
loss_results_synt = list(results["results_synt"]["results"]["loss"].values())

times_results_image = list(results["results_image"]["results"]["times"].values())
scores_results_image = list(results["results_image"]["results"]["scores"].values())
loss_results_image = list(results["results_image"]["results"]["loss"].values())
reg_list = results["reg_list"]["reg_list"]


fig, axs = plt.subplots(1, 2)


new_times = np.linspace(0, 300, 100)
recoveries = []
for i in range(len(times_results_image)):
    recoveries.append(np.zeros((len(times_results_image[i]), len(new_times))))

t_max = np.max(new_times)
for i in range(len(times_results_image)):
    for j in range(len(times_results_image[i])):
        if times_results_image[i][j][-1] < t_max:
            times_results_image[i][j][-1] = t_max
        f = interpolate.interp1d(times_results_image[i][j], scores_results_image[i][j])
        recoveries[i][j] = f(new_times)

recoveries = np.array(recoveries)
max_rec = np.max(recoveries[:, :, -1], axis=0)[None, :, None] + 1e-3
recoveries = max_rec - recoveries

for i in range(recoveries.shape[0]):
    if i == 0:
        axs[0].plot(
            new_times,
            recoveries[i].mean(axis=0),
            label="Unscaled"
        )
        axs[0].fill_between(
            new_times,
            recoveries[i].mean(axis=0) + recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            recoveries[i].mean(axis=0) - recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            alpha=0.2
        )
    elif i == 1:
        axs[0].plot(
            new_times,
            recoveries[i].mean(axis=0),
            label="Scaled"
        )
        axs[0].fill_between(
            new_times,
            recoveries[i].mean(axis=0) + recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            recoveries[i].mean(axis=0) - recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            alpha=0.2
        )
    else:
        axs[0].plot(
            new_times,
            recoveries[i].mean(axis=0),
            label=f"Scaled + reg {round(reg_list[i-2], 2)}"
        )
        axs[0].fill_between(
            new_times,
            recoveries[i].mean(axis=0) + recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            recoveries[i].mean(axis=0) - recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            alpha=0.2
        )
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Max. score - score")
axs[0].set_title("Image")
axs[0].set_ylim([1e-2, 0.3])
axs[0].set_xlim([20, 250])
axs[0].set_xticks([20, 60, 100, 140, 180, 220])
axs[0].grid()



new_times = np.linspace(0, 20, 100)
recoveries = []
for i in range(len(times_results_synt)):
    recoveries.append(np.zeros((len(times_results_synt[i]), len(new_times))))

t_max = np.max(new_times)
for i in range(len(times_results_synt)):
    for j in range(len(times_results_synt[i])):
        if times_results_synt[i][j][-1] < t_max:
            times_results_synt[i][j][-1] = t_max
        f = interpolate.interp1d(times_results_synt[i][j], scores_results_synt[i][j])
        recoveries[i][j] = f(new_times)

recoveries = np.array(recoveries)
max_rec = 1
recoveries = max_rec - recoveries

for i in range(recoveries.shape[0]):
    if i == 0:
        axs[1].plot(
            new_times,
            recoveries[i].mean(axis=0),
            label="Unscaled"
        )
        axs[1].fill_between(
            new_times,
            recoveries[i].mean(axis=0) + recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            recoveries[i].mean(axis=0) - recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            alpha=0.2
        )
    elif i == 1:
        axs[1].plot(
            new_times,
            recoveries[i].mean(axis=0),
            label="Scaled"
        )
        axs[1].fill_between(
            new_times,
            recoveries[i].mean(axis=0) + recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            recoveries[i].mean(axis=0) - recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            alpha=0.2
        )
    else:
        axs[1].plot(
            new_times,
            recoveries[i].mean(axis=0),
            label=f"Scaled + reg {round(reg_list[i-2], 2)}"
        )
        axs[1].fill_between(
            new_times,
            recoveries[i].mean(axis=0) + recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            recoveries[i].mean(axis=0) - recoveries[i].std(axis=0) * 2 / recoveries.shape[1],
            alpha=0.2
        )
axs[1].set_yscale("log")
axs[1].set_xlabel("Time (s)")
axs[1].set_title("Sensing")
axs[1].set_xlim([5, 15])
axs[1].set_xticks([5, 7, 9, 11, 13, 15])
axs[1].set_ylim([1e-3, 1e-1])
axs[1].grid()

plt.legend(bbox_to_anchor=(1.05, 1.2), loc="upper left")
plt.savefig("../figures/gradient_scaling.pdf")
