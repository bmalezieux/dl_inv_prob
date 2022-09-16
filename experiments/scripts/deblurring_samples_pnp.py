import datetime
from dl_inv_prob.common_utils import (
    np_to_torch,
    pil_to_np,
    torch_to_np,
)
from dl_inv_prob.dataset import (
    DenoisingDataset,
    DenoisingDeblurringDataset,
    train_val_dataloaders,
)
from dl_inv_prob.models.DnCNN import DnCNN
from dl_inv_prob.training import train
from dl_inv_prob.utils import psnr, gaussian_kernel, determinist_blurr
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import time
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import itertools

start_time = time.time()

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
TRAINING = True

# Paths

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(EXPERIMENTS, "data")
CLEAN_DATA = os.path.join(DATA_PATH, "Train400")
IMG_PATH = os.path.join(DATA_PATH, "flowers.png")
MODELS_PATH = os.path.join(EXPERIMENTS, "../dl_inv_prob/models/pretrained")
CORRUPTED_DENOISER_NAME = "DnCNN_denoising_deblurring"
CLEAN_DENOISER_NAME = "DnCNN_denoising"
RESULTS = os.path.join(EXPERIMENTS, "results/deblurring_pnp.csv")
SIZE = 180

# Hyperparameters

hyperparams = {
    "sigma_max_denoiser": [0.05, 0.1, 0.3],
    "sigma_sample": [0, 0.02, 0.05, 0.1],
    "sigma_blurr": np.arange(0.1, 1.0, 0.1),
    "size_blurr": [10],
    "n_samples": [200],
    "n_epochs": [50],
    "batch_size": [32],
    "lr": [0.001],
    "step_pnp": [1.0],
    "iter_pnp": [100]
}

keys, values = zip(*hyperparams.items())
permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

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

# Load test image
img = Image.open(IMG_PATH)
img = img.convert("L")
img_np = pil_to_np(img)
img = np_to_torch(img_np).type(DTYPE).to(DEVICE)

mse = nn.MSELoss()
dict_results = {}

for params in permuts_params:

    # Corrupt the test image with a blurr and noise
    img, corrupted, blurr = determinist_blurr(
        IMG_PATH,
        params["sigma_blurr"],
        params["size_blurr"],
        params["sigma_sample"],
        size=SIZE,
        dtype=DTYPE,
        device=DEVICE
    )

    y_conv_display = F.conv2d(
        img[None, :, :],
        torch.flip(blurr[None, :, :], dims=[2, 3]),
        padding="same"
    )
    noise_same = torch.randn(y_conv_display.shape, generator=RNG, dtype=DTYPE, device=DEVICE)

    img = img[None, :, :]
    blurr = blurr[None, :, :]
    corrupted_img = corrupted[None, :, :]
    corrupted_img_np = torch_to_np(corrupted_img)

    img_np = torch_to_np(img)
    corrupted_img_same = y_conv_display + params["sigma_sample"] * noise_same
    corrupted_img_same_np = torch_to_np(corrupted_img_same)

    # Store psnr of the corrupted image
    psnr_corr = psnr(corrupted_img_same_np, img_np)

    # Create inpainted denoising dataset
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
    file_name = "DnCNN_pnp_deblurring"

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

        # Train the denoising network on the dataset
        type = "clean" if mode == "denoising" else "corrupted"
        print(f"Training the denoising network on the {type} dataset")

        denoiser, losses, val_losses = train(
            model=denoiser,
            loss_fn=mse,
            mode=mode,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            file_name=file_name,
            model_path=MODELS_PATH,
            learning_rate=params["lr"],
            device=DEVICE,
            n_epochs=params["n_epochs"],
        )

        # Load the trained denoiser
        denoiser = DnCNN().to(DEVICE).type(DTYPE)
        DENOISER_PATH = os.path.join(MODELS_PATH, file_name + ".pt")
        denoiser.load_state_dict(torch.load(DENOISER_PATH))
        denoiser.eval()

        # Reconstruct the image with Plug-and-Play Forward-Backward
        type = "clean" if mode == "denoising" else "corrupted"
        print(f"PnP with the {type} denoiser")

        out = torch.zeros_like(img, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            for iter in range(1, params["iter_pnp"] + 1):
                grad = F.conv2d(
                    F.conv_transpose2d(out, blurr) - corrupted_img,
                    blurr
                )
                out -= params["step_pnp"] * grad
                out = torch.clip(denoiser(out), 0, 1)
                loss = mse(F.conv_transpose2d(out, blurr), corrupted_img)

        out_np = torch_to_np(out)

        # Store PSNR
        psnr_rec = psnr(out_np, img_np)
        if mode == "denoising_deblurring":
            psnr_rec_unsupervised = psnr_rec
        elif mode == "denoising":
            psnr_rec_supervised = psnr_rec

        print(f"psnr_rec = {psnr_rec:.2f}, mode={mode}")
        print(f"psnr_corr = {psnr_corr:.2f}")

    # Saving results
    results = {
        "psnr_rec_unsupervised": psnr_rec_unsupervised,
        "psnr_rec_supervised": psnr_rec_supervised,
        "psnr_corr": psnr_corr
    }

    for key in params.keys():
        if key not in dict_results:
            dict_results[key] = [params[key]]
        else:
            dict_results[key].append(params[key])

    for key in results.keys():
        if key not in dict_results:
            dict_results[key] = [results[key]]
        else:
            dict_results[key].append(results[key])


# Save the results
results_df = pd.DataFrame(dict_results)
results_df.to_csv(str(RESULTS))
