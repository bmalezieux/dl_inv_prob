# %%
import datetime
from dl_inv_prob.common_utils import (
    np_to_torch,
    pil_to_np,
    torch_to_np,
)
from dl_inv_prob.dataset import (
    DenoisingDataset,
    DenoisingInpaintingDataset,
    train_val_dataloaders,
)
from dl_inv_prob.models.DnCNN import DnCNN
from dl_inv_prob.training import train
from dl_inv_prob.utils import psnr, determinist_inpainting
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import time
import torch
import torch.optim
import torch.nn as nn
from pathlib import Path
import itertools
from tqdm import tqdm

start_time = time.time()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
TRAINING = True

# Paths

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(EXPERIMENTS, "data")
CLEAN_DATA = os.path.join(DATA_PATH, "imagewoof_resized")
TEST_DATA = os.path.join(DATA_PATH, "imagewoof_resized_test")
MODELS_PATH = os.path.join(EXPERIMENTS, "../dl_inv_prob/models/pretrained")
RESULTS = os.path.join(EXPERIMENTS, "results/inpainting_pnp_number_samples_woof.csv")

# Hyperparameters

# %%
params = {
    "sigma_max_denoiser": 0.05,
    "sigma_sample": 0.,
    "prop": 0.1,
    "n_samples": 1000,
    "n_epochs": 40,
    "batch_size": 32,
    "lr": 1e-3,
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

# %%

# Create inpainted denoising dataset
corrupted_dataset = DenoisingInpaintingDataset(
    path=CLEAN_DATA,
    n_samples=params["n_samples"],
    noise_std=params["sigma_max_denoiser"],
    rng=RNG,
    prop=params["prop"],
    sigma=params["sigma_sample"],
    dtype=DTYPE,
    device=DEVICE,
)

file_name = "DnCNN_pnp_inpainting_example_woof"

train_dataloader, val_dataloader = train_val_dataloaders(
    dataset=corrupted_dataset,
    train_batch_size=params["batch_size"],
    val_batch_size=params["batch_size"],
    val_split_fraction=0.1,
    np_rng=NP_RNG,
    seed=SEED,
    num_workers=0,
)

denoiser = DnCNN().to(DEVICE).type(DTYPE)

# %%

for i in range(10):

    print("Training the network")
    denoiser, losses, val_losses = train(
        model=denoiser,
        loss_fn=mse,
        mode="denoising_inpainting",
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        file_name=file_name,
        model_path=MODELS_PATH,
        learning_rate=params["lr"],
        device=DEVICE,
        n_epochs=10,
        show_every=10,
    )

    N_TEST = 50
    # Load the trained denoiser
    # denoiser = DnCNN().to(DEVICE).type(DTYPE)
    # DENOISER_PATH = os.path.join(MODELS_PATH, file_name + ".pt")
    # denoiser.load_state_dict(torch.load(DENOISER_PATH))
    denoiser.eval()

    # Reconstruct the image with Plug-and-Play Forward-Backward
    print("Reconstruction with PnP")
    psnr_rec_unsupervised = 0

    for filename in tqdm(os.listdir(TEST_DATA)[:N_TEST]):
        img_test, corrupted_img_test, mask_test = determinist_inpainting(
            img_path=os.path.join(TEST_DATA, filename),
            prop=params["prop"],
            sigma=params["sigma_sample"],
            seed=SEED,
            dtype=DTYPE,
            device=DEVICE,
        )
        mask_test = mask_test[None, :, :]
        img_test = img_test[None, :, :]
        corrupted_img_test = corrupted_img_test[None, :, :]
        corrupted_img_test_np = torch_to_np(corrupted_img_test)
        img_test_np = torch_to_np(img_test)

        out = torch.zeros_like(img_test, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            for iter in range(1, params["iter_pnp"] + 1):
                grad = mask_test * (out - corrupted_img_test)
                out -= params["step_pnp"] * grad
                out = torch.clip(denoiser(out), 0, 1)
                loss = mse(out * mask_test, corrupted_img_test)

        out_np = torch_to_np(out)

        # Store PSNR
        psnr_rec = psnr(out_np, img_test_np)
        psnr_rec_unsupervised += psnr_rec

    psnr_rec_unsupervised /= N_TEST

    print(f"psnr_rec_unsupervised = {psnr_rec_unsupervised:.2f}")


# %%
