from dl_inv_prob.utils import create_image_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('figures_style_full.mplstyle')

df = pd.read_pickle('../results/inpainting_conv_digits.pickle')
scores = df.scores.scores
sizes = df.sizes.sizes
s_values = df.s_values.s_values
img = create_image_digits(100, 50, k=0.1)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(img, cmap='gray')
ax1.axis('off')

for i in range(scores.shape[1]):
    ax2.plot(sizes, scores[:, i, :].mean(axis=0), '-',
             label=f's={s_values[i]:.2f}')
    ax2.fill_between(
            sizes,
            np.quantile(scores[:, i, :], 0.1, axis=0),
            np.quantile(scores[:, i, :], 0.9, axis=0),
            alpha=0.2
        )
ax2.set_xlabel('Image size')
ax2.set_ylabel('Score')
ax2.grid()
ax2.set_title('Digits recovery')
ax2.legend(fontsize=5)

plt.savefig("../figures/inpainting_conv_digits.pdf")
