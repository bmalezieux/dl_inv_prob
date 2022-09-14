
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
RESULTS = os.path.join(EXPERIMENTS, "results", "deblurring_color")
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
img = Image.open(IMG).convert("RGB").resize((SIZE, SIZE), Image.ANTIALIAS)
img = pil_to_np(img)

os.makedirs(RESULTS, exist_ok=True)
plt.imsave(os.path.join(RESULTS, "clean.png"), np.transpose(img, (1, 2, 0)))

# %%
y = np.array(img)
y = torch.tensor(y, device=DEVICE, dtype=torch.float, requires_grad=False)
kernel = gaussian_kernel(10, 0.3)
kernel = torch.tensor(kernel, device=DEVICE, dtype=torch.float,
                      requires_grad=False)
y_conv = F.conv_transpose2d(y[:, None, :, :], kernel[None, None, :, :])
y_conv_display = F.conv2d(
    y[:, None, :, :],
    torch.flip(kernel[None, None, :, :], dims=[2, 3]),
    padding="same"
)

y_conv = y_conv.detach().cpu().numpy().squeeze()
y_conv_display = y_conv_display.detach().cpu().numpy().squeeze()
kernel = kernel.detach().cpu().numpy().squeeze()
y = y.detach().cpu().numpy().squeeze()

plt.imsave(
    os.path.join(RESULTS, "blurred.png"),
    np.transpose(y_conv_display, (1, 2, 0))
)

# %%

# Models
# DIP
model = "SkipNet"
n_iter = 1000
lr = 0.01
sigma_input_noise = 1.0
sigma_reg_noise = 0.03
input_depth = 32

dip = DIPDeblurring(
    model=model,
    n_iter=n_iter,
    lr=lr,
    sigma_input_noise=sigma_input_noise,
    sigma_reg_noise=sigma_reg_noise,
    input_depth=input_depth
)

# Wavelets
lambd = 0.01
sparse = SparseWaveletsDeblurring(lambd=lambd)

# TV
lambd = 0.01
proxtv = ProxTVDeblurring(lambd=lambd)

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

models = {
    # "tv": proxtv,
    # "wavelets": sparse,
    # "cdl": cdl,
    "dip": dip,
}

# %%

for model in models:
    print(model)
    result = np.zeros(y.shape)
    for i in range(3):
        print(i)
        models[model].fit(y_conv[i][None, :, :], kernel[None, :, :])
        result[i] = np.clip(models[model].rec(), 0, 1)
    plt.imsave(
        os.path.join(RESULTS, f"{model}.png"),
        np.transpose(result, (1, 2, 0))
    )

# %%
