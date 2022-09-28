# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from dl_inv_prob.utils import create_image_digits


# %%
plt.style.use('figures_style_small.mplstyle')

df = pd.read_csv('../results/inpainting_cdl_digits_score.csv')

# %%
constant_color = 3
c = np.arange(1, len(pd.unique(df["rho"])) + constant_color)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

img = create_image_digits(100, 50, k=0.1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.2, 3))

ax1.imshow(img, cmap='gray')
ax1.axis('off')

for i, rho in enumerate(pd.unique(df["rho"])):
    scores = []
    for size in pd.unique(df["size"]):
        current_score = df[(df["size"] == size)
                           & (df["rho"] == rho)
                           & (df["sigma"] == 0.1)]["score"].max()
        scores.append(current_score)
    ax2.plot(pd.unique(df["size"]), scores, label=rho,
             color=cmap.to_rgba(i + constant_color))

ax2.set_xlabel('Image size')
ax2.set_ylabel('Score')
ax2.grid()
ax2.set_title('Digits recovery')
ax2.legend(fontsize=5, loc=4, title="rho")

plt.tight_layout()
plt.savefig("../figures/inpainting_conv_digits.pdf")

# %%
