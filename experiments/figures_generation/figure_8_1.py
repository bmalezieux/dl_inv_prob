# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('figures_style_small.mplstyle')

# %%

fig, axs = plt.subplots(2, 2, figsize=(4, 4))

axs[0, 0].imshow(plt.imread("../results/deblurring_pnp/psd_original.png"), cmap="gray")
axs[0, 0].axis("off")
axs[0, 0].set_title("Original", fontsize=12)

axs[0, 1].imshow(plt.imread("../results/deblurring_pnp/psd_start.png"), cmap="gray")
axs[0, 1].axis("off")
axs[0, 1].set_title("Blurred", fontsize=12)

axs[1, 0].imshow(plt.imread("../results/deblurring_pnp/psd_supervised.png"), cmap="gray")
axs[1, 0].axis("off")
axs[1, 0].set_title("DnCNN: clean data", fontsize=12)

axs[1, 1].imshow(plt.imread("../results/deblurring_pnp/psd_unsupervised.png"), cmap="gray")
axs[1, 1].axis("off")
axs[1, 1].set_title("Original")
axs[1, 1].set_title("DnCNN: blurred data", fontsize=12)

plt.tight_layout()
plt.savefig("../figures/deblurring_pnp_psd.pdf")
# %%
