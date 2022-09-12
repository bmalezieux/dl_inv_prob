# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl_inv_prob.utils import create_image_digits


# %%
plt.style.use('figures_style_small.mplstyle')

df = pd.read_csv('../results/inpainting_cdl_digits_score.csv')

# %%
img = create_image_digits(100, 50, k=0.1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.2, 3))

ax1.imshow(img, cmap='gray')
ax1.axis('off')

for rho in pd.unique(df["rho"]):
    scores = []
    for size in pd.unique(df["size"]):
        current_score = df[(df["size"] == size) & (df["rho"] == rho) & (df["sigma"] == 0.1)]["score"].max()
        scores.append(current_score)
    ax2.plot(pd.unique(df["size"]), scores, label=rho)

ax2.set_xlabel('Image size')
ax2.set_ylabel('Score')
ax2.grid()
ax2.set_title('Digits recovery')
ax2.legend(fontsize=5, loc=4, title="rho")

plt.tight_layout()
plt.savefig("../figures/inpainting_conv_digits.pdf")

# %%
