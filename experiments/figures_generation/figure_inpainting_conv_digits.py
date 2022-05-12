import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('figures_style_full.mplstyle')

df = pd.read_pickle('../results/inpainting_conv_digits.pickle')
scores = df.scores.scores
sizes = df.scores.scores
s_values = df.s_values.s_values

for i in range(scores.shape[1]):
    plt.plot(sizes, scores[:, i, :].mean(axis=0), '-o',
             label=f's={s_values[i]}')
    plt.fill_between(
            sizes,
            np.quantile(scores[:, i, :], 0.1, axis=0),
            np.quantile(scores[:, i, :], 0.9, axis=0),
            alpha=0.2
        )
plt.xlabel('Image size')
plt.ylabel('Score')
plt.grid()
plt.title('Digits recovery')
plt.legend()
plt.show()
