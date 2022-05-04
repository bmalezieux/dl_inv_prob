import numpy as np
import matplotlib.pyplot as plt


plt.style.use('figures_style_small.mplstyle')

new_times = np.load("../results/scaling_gradient_times.npy")
recoveries = np.load("../results/scaling_gradient_recoveries.npy")
reg_list = np.load("../results/scaling_gradient_reg_list.npy")

max_rec = np.max(recoveries)
for i in range(recoveries.shape[0]):
    if i == 0:
        plt.plot(
            new_times,
            max_rec - recoveries[i].mean(axis=0),
            label="Unscaled"
        )
    elif i == 1:
        plt.plot(
            new_times,
            max_rec - recoveries[i].mean(axis=0),
            label="Scaled"
        )
    else:
        plt.plot(
            new_times,
            max_rec - recoveries[i].mean(axis=0),
            label=f"Scaled + reg {round(reg_list[i-2], 2)}"
        )
plt.yscale("log")
plt.xlabel("Time (s)")
plt.ylabel("S* - S")
plt.legend()
plt.savefig("../figures/gradient_scaling.pdf")
