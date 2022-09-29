# %%
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from pathlib import Path


try:
    os.mkdir("../figures")
except OSError:
    pass


def psnr(im, imref, d=255):
    mse = np.mean((im - imref)**2)
    return 10 * np.log(d * d / mse) / np.log(10)


def compute_psd(image):

    mag = 0
    for i in range(3):
        f = np.fft.fft2(np.array(image)[:, :, i])
        fshift = np.fft.fftshift(f)
        mag += 20 * np.log(np.abs(fshift))

    return mag / 3.


noise_level = 0

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results", "deblurring_color")
FIGURES = os.path.join(EXPERIMENTS, "figures")
original = Image.open(os.path.join(RESULTS, "clean.png"))
degraded = Image.open(os.path.join(RESULTS, "blurred.png"))
tv = Image.open(os.path.join(RESULTS, "tv.png"))
wavelets = Image.open(os.path.join(RESULTS, "wavelets.png"))
cdl = Image.open(os.path.join(RESULTS, "cdl.png"))
dip = Image.open(os.path.join(RESULTS, "dip.png"))

print(f"cdl : {psnr(np.array(original), np.array(cdl))}")
print(f"dip : {psnr(np.array(original), np.array(dip))}")
print(f"wavelets: {psnr(np.array(original), np.array(wavelets))}")
print(f"TV : {psnr(np.array(original), np.array(tv))}")

fig, axs = plt.subplots(2, 6)

# Original image
axs[0, 0].imshow(np.array(original), vmin=0, vmax=255)
axins = axs[0, 0].inset_axes([0.6, 0, 0.4, 0.4])
axins.imshow(np.array(original))
x1, x2, y1, y2 = 160, 200, 160, 200
axins.set_xlim(x1, x2)
axins.set_ylim(y2, y1)
axins.set_xticklabels('')
axins.set_yticklabels('')
axs[0, 0].indicate_inset_zoom(axins, edgecolor="black")
axs[0, 0].set_title("Original", fontsize=20)
axs[0, 0].set_axis_off()

# Degradation
axs[0, 1].imshow(np.array(degraded), vmin=0, vmax=255)
axins = axs[0, 1].inset_axes([0.6, 0, 0.4, 0.4])
axins.imshow(np.array(degraded))
x1, x2, y1, y2 = 160, 200, 160, 200
axins.set_xlim(x1, x2)
axins.set_ylim(y2, y1)
axins.set_xticklabels('')
axins.set_yticklabels('')
axs[0, 1].indicate_inset_zoom(axins, edgecolor="black")
axs[0, 1].set_title("Observation", fontsize=20)
axs[0, 1].set_axis_off()

# CDL
axs[0, 2].imshow(np.array(cdl), vmin=0, vmax=255)
axins = axs[0, 2].inset_axes([0.6, 0, 0.4, 0.4])
axins.imshow(np.array(cdl))
x1, x2, y1, y2 = 160, 200, 160, 200
axins.set_xlim(x1, x2)
axins.set_ylim(y2, y1)
axins.set_xticklabels('')
axins.set_yticklabels('')
axs[0, 2].indicate_inset_zoom(axins, edgecolor="black")
axs[0, 2].set_title("CDL", fontsize=20)
axs[0, 2].set_axis_off()

# DIP
axs[0, 3].imshow(np.array(dip), vmin=0, vmax=255)
axins = axs[0, 3].inset_axes([0.6, 0, 0.4, 0.4])
axins.imshow(np.array(dip))
x1, x2, y1, y2 = 160, 200, 160, 200
axins.set_xlim(x1, x2)
axins.set_ylim(y2, y1)
axins.set_xticklabels('')
axins.set_yticklabels('')
axs[0, 3].indicate_inset_zoom(axins, edgecolor="black")
axs[0, 3].set_title("DIP", fontsize=20)
axs[0, 3].set_axis_off()

# Wavelets
axs[0, 4].imshow(np.array(wavelets), vmin=0, vmax=255)
axins = axs[0, 4].inset_axes([0.6, 0, 0.4, 0.4])
axins.imshow(np.array(wavelets))
x1, x2, y1, y2 = 160, 200, 160, 200
axins.set_xlim(x1, x2)
axins.set_ylim(y2, y1)
axins.set_xticklabels('')
axins.set_yticklabels('')
axs[0, 4].indicate_inset_zoom(axins, edgecolor="black")
axs[0, 4].set_title("Wavelets", fontsize=20)
axs[0, 4].set_axis_off()

# TV
axs[0, 5].imshow(np.array(tv), vmin=0, vmax=255)
axins = axs[0, 5].inset_axes([0.6, 0, 0.4, 0.4])
axins.imshow(np.array(tv))
x1, x2, y1, y2 = 160, 200, 160, 200
axins.set_xlim(x1, x2)
axins.set_ylim(y2, y1)
axins.set_xticklabels('')
axins.set_yticklabels('')
axs[0, 5].indicate_inset_zoom(axins, edgecolor="black")
axs[0, 5].set_title("TV", fontsize=20)
axs[0, 5].set_axis_off()


# PSD

axs[1, 0].imshow(compute_psd(original), cmap="gray")
axs[1, 0].set_axis_off()

axs[1, 1].imshow(compute_psd(degraded), cmap="gray")
axs[1, 1].set_axis_off()

axs[1, 2].imshow(compute_psd(cdl), cmap="gray")
axs[1, 2].set_axis_off()

axs[1, 3].imshow(compute_psd(dip), cmap="gray")
axs[1, 3].set_axis_off()

axs[1, 4].imshow(compute_psd(wavelets), cmap="gray")
axs[1, 4].set_axis_off()

axs[1, 5].imshow(compute_psd(tv), cmap="gray")
axs[1, 5].set_axis_off()


fig.set_figwidth(15)
fig.set_figheight(6)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "deblurring_color_zoom.pdf"))
# %%
