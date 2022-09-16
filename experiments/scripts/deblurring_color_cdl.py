
# %%
import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dl_inv_prob.common_utils import (
    pil_to_np,
)
from dl_inv_prob.dl import Deconvolution
from dl_inv_prob.utils import gaussian_kernel
from dl_inv_prob.dip import DIPDeblurring
from utils.tv import ProxTVDeblurring
from utils.wavelets import SparseWaveletsDeblurring
from PIL import Image
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = os.path.join(EXPERIMENTS, "data")
IMG = os.path.join(DATA, "flowers.png")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Reproducibility
SEED = 2022
NP_RNG = np.random.default_rng(SEED)
RNG = torch.Generator(device=DEVICE)
RNG.manual_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.use_deterministic_algorithms(True)

SIZE = 256
img = Image.open(IMG).convert("L").resize((SIZE, SIZE), Image.ANTIALIAS)
img = pil_to_np(img)

# %%
y = np.array(img)
y = torch.tensor(y, device=DEVICE, dtype=torch.float, requires_grad=False)
kernel = gaussian_kernel(10, 0.3)
kernel = torch.tensor(kernel, device=DEVICE, dtype=torch.float,
                      requires_grad=False)
y_conv = F.conv_transpose2d(y[None, :, :], kernel[None, None, :, :])
y_conv_display = F.conv2d(
    y[None, :, :],
    torch.flip(kernel[None, None, :, :], dims=[2, 3]),
    padding="same"
)

y_conv = y_conv.detach().cpu().numpy().squeeze()
y_conv_display = y_conv_display.detach().cpu().numpy().squeeze()
kernel = kernel.detach().cpu().numpy().squeeze()
y = y.detach().cpu().numpy().squeeze()


# %%

# CDL
dim_patch = 20
n_atoms = 50
cdl = Deconvolution(
        n_atoms,
        init_D=None,
        device=DEVICE,
        rng=NP_RNG,
        atom_height=dim_patch,
        atom_width=dim_patch,
        lambd=0.1
    )

cdl.fit(y[None, :, :], np.array([1])[None, None, None, :])

# %%
result_cdl_supervised_50 = np.clip(cdl.rec(y_conv[None, :, :], kernel[None, None, :, :]), 0, 1)

# %%

# CDL
dim_patch = 20
n_atoms = 10
cdl = Deconvolution(
        n_atoms,
        init_D=None,
        device=DEVICE,
        rng=NP_RNG,
        atom_height=dim_patch,
        atom_width=dim_patch,
        lambd=0.1
    )

cdl.fit(y[None, :, :], np.array([1])[None, None, None, :])

# %%
result_cdl_supervised_10 = np.clip(cdl.rec(y_conv[None, :, :], kernel[None, None, :, :]), 0, 1)

# %%
# CDL
dim_patch = 20
n_atoms = 10
cdl = Deconvolution(
        n_atoms,
        init_D=None,
        device=DEVICE,
        rng=NP_RNG,
        atom_height=dim_patch,
        atom_width=dim_patch,
        lambd=0.1
    )

cdl.fit(y_conv[None, :, :], kernel[None, None, :, :])

# %%
result_cdl_unsupervised_true = np.clip(cdl.rec(), 0, 1)

# %%
def compute_psd(image):

    f = np.fft.fft2(np.array(image))
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift))

    return mag


fig, axs = plt.subplots(2, 5, figsize=(15, 6))

axs[0, 0].imshow(y, cmap="gray")
axs[0, 0].set_title("Original")
axs[0, 0].set_axis_off()
axs[1, 0].imshow(compute_psd(y), cmap="gray")
axs[1, 0].set_axis_off()

axs[0, 1].imshow(y_conv_display, cmap="gray")
axs[0, 1].set_title("Blurred")
axs[0, 1].set_axis_off()
axs[1, 1].imshow(compute_psd(y_conv_display), cmap="gray")
axs[1, 1].set_axis_off()

axs[0, 2].imshow(result_cdl_unsupervised[0, 0], cmap="gray")
axs[0, 2].set_title("Supervised 50 atoms")
axs[0, 2].set_axis_off()
axs[1, 2].imshow(compute_psd(result_cdl_unsupervised[0, 0]), cmap="gray")
axs[1, 0].set_axis_off()

axs[0, 3].imshow(result_cdl_supervised_10[0, 0], cmap="gray")
axs[0, 3].set_title("Supervised 10 atoms")
axs[0, 3].set_axis_off()
axs[1, 3].imshow(compute_psd(result_cdl_supervised_10[0, 0]), cmap="gray")
axs[1, 3].set_axis_off()

axs[0, 4].imshow(result_cdl_unsupervised_true[0, 0], cmap="gray")
axs[0, 4].set_title("Unsupervised")
axs[0, 4].set_axis_off()
axs[1, 4].imshow(compute_psd(result_cdl_unsupervised_true[0, 0]), cmap="gray")
axs[1, 4].set_axis_off()
plt.savefig("../figures/example_deblurring_cdl.png")
# %%
from dl_inv_prob.utils import psnr, is_divergence


print(psnr(result_cdl_unsupervised[0, 0], y))
print(psnr(result_cdl_supervised_10[0, 0], y))
print(psnr(result_cdl_unsupervised_true[0, 0], y))


print(is_divergence(result_cdl_unsupervised[0, 0], y))
print(is_divergence(result_cdl_supervised_10[0, 0], y))
print(is_divergence(result_cdl_unsupervised_true[0, 0], y))
# %%
