# %%
from dl_inv_prob.common_utils import (
    torch_to_np,
)
from dl_inv_prob.dataset import (
    DenoisingDataset,
    DenoisingDeblurringDataset,
    train_val_dataloaders,
)
from dl_inv_prob.models.DnCNN import DnCNN
from dl_inv_prob.training import train
from dl_inv_prob.utils import psnr, determinist_blurr, split_psnr
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import itertools
from tqdm import tqdm

import matplotlib.pyplot as plt



DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Paths

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(EXPERIMENTS, "data")
CLEAN_DATA = os.path.join(DATA_PATH, "imagewoof_resized")
TEST_DATA = os.path.join(DATA_PATH, "imagewoof_resized_test")
RESULTS = os.path.join(EXPERIMENTS, "results/deblurring_pnp")
os.makedirs(RESULTS, exist_ok=True)
N_TEST = 50

# Hyperparameters

params = {
    "sigma_max_denoiser": 0.05,
    "sigma_sample": 0.02,
    "sigma_blurr": 0.3,
    "size_blurr": 10,
    "n_samples": 1000,
    "n_epochs": 30,
    "batch_size": 32,
    "lr": 0.001,
    "step_pnp": 1.0,
    "iter_pnp": 100
}

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
np.random.seed(SEED)
random.seed(SEED)

mse = nn.MSELoss()
dict_results = {}


def compute_psd(image):

    f = np.fft.fft2(np.array(image))
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift))

    return mag


# Create clean denoising dataset
corrupted_dataset = DenoisingDeblurringDataset(
    path=CLEAN_DATA,
    n_samples=params["n_samples"],
    noise_std=params["sigma_max_denoiser"],
    rng=RNG,
    sigma_blurr=params["sigma_blurr"],
    size_blurr=params["size_blurr"],
    sigma=params["sigma_sample"],
    dtype=DTYPE,
    device=DEVICE,
)

# Create clean denoising dataset
clean_dataset = DenoisingDataset(
    path=CLEAN_DATA,
    n_samples=params["n_samples"],
    noise_std=params["sigma_max_denoiser"],
    rng=RNG,
    dtype=DTYPE,
    device=DEVICE,
)

datasets = [corrupted_dataset, clean_dataset]
modes = ["denoising_deblurring", "denoising"]

psd_supervised = 0
psd_unsupervised = 0

for dataset, mode in zip(datasets, modes):

    train_dataloader, val_dataloader = train_val_dataloaders(
        dataset=dataset,
        train_batch_size=params["batch_size"],
        val_batch_size=params["batch_size"],
        val_split_fraction=0.1,
        np_rng=NP_RNG,
        seed=SEED,
        num_workers=0,
    )

    denoiser = DnCNN().to(DEVICE).type(DTYPE)

    denoiser, losses, val_losses = train(
        model=denoiser,
        loss_fn=mse,
        mode=mode,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=params["lr"],
        device=DEVICE,
        n_epochs=5,
    )

    denoiser.eval()

    psd_start = 0
    psd_original = 0

    for filename in os.listdir(TEST_DATA)[:N_TEST]:
        img_test, corrupted_img_test, blurr_test = determinist_blurr(
            os.path.join(TEST_DATA, filename),
            params["sigma_blurr"],
            params["size_blurr"],
            params["sigma_sample"],
            dtype=DTYPE,
            device=DEVICE
        )
        y_conv_display = F.conv2d(
            img_test[None, :, :],
            torch.flip(blurr_test[None, :, :], dims=[2, 3]),
            padding="same"
        )
        noise_same = torch.randn(y_conv_display.shape, generator=RNG, dtype=DTYPE, device=DEVICE)

        img_test = img_test[None, :, :]
        blurr_test = blurr_test[None, :, :]
        corrupted_img_test = corrupted_img_test[None, :, :]
        corrupted_img_test_np = torch_to_np(corrupted_img_test)
        img_test_np = torch_to_np(img_test)
        blurr_test_np = torch_to_np(blurr_test)

        out = corrupted_img_test.clone()
        for i in range(10):
            out = denoiser(out)

        if mode == "denoising_deblurring":
            psd_unsupervised += compute_psd(torch_to_np(out))
        else:
            psd_supervised += compute_psd(torch_to_np(out))
        psd_start += compute_psd(corrupted_img_test_np)
        psd_original += compute_psd(img_test_np)

    if mode == "denoising_deblurring":
        psd_unsupervised = psd_unsupervised.squeeze() / N_TEST
    else:
        psd_supervised = psd_supervised.squeeze() / N_TEST
    psd_start = psd_start.squeeze() / N_TEST
    psd_original = psd_original.squeeze() / N_TEST

# %%

psds = [psd_supervised, psd_unsupervised, psd_start, psd_original]
names = ["psd_supervised", "psd_unsupervised", "psd_start", "psd_original"]

for psd, name in zip(psds, names):
    plt.imshow(psd, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"{name}.png"), bbox_inches="tight", pad_inches=0)
    plt.clf()

# %%




# fig, axs = plt.subplots(1, 3, figsize=(10, 3))

# axs[0].imshow(psd_original, cmap="gray")
# axs[0].set_title("Original")
# axs[0].axis("off")

# axs[1].imshow(psd_start, cmap="gray")
# axs[1].set_title("Blurred")
# axs[1].axis("off")

# axs[2].imshow(psd_result, cmap="gray")
# axs[2].set_title("Result")
# axs[2].axis("off")

# plt.tight_layout()
# plt.savefig(RESULTS)
# %%
