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
from dl_inv_prob.utils import psnr
from math import floor
import numpy as np
import os
import pandas as pd
from PIL import Image
import time
import torch
import torch.optim
import torch.nn as nn

start_time = time.time()

DEVICE = "cuda:2"
DTYPE = torch.float32
TRAINING = True

# Paths
DATA_PATH = "experiments/data"
CLEAN_DATA = os.path.join(DATA_PATH, "Train400")
IMG_PATH = os.path.join(DATA_PATH, "flowers.png")
MODELS_PATH = "dl_inv_prob/models/pretrained"
CORRUPTED_DENOISER_NAME = "DnCNN_denoising_inpainting"
CLEAN_DENOISER_NAME = "DnCNN_denoising"

# Hyperparameters
N_EPOCHS = 100
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
LR = 0.001  # Learning rate to train the denoiser
N_SAMPLES = [100, 400]  # Training samples
SIGMA_MAX_DENOISER = 0.3  # Max noise level to train the denoiser
SIGMA_TRAIN_SAMPLE = 0.0  # Noise level of "clean" training samples
SIGMA_TEST_SAMPLE = 0.1  # Noise level of the corrupted test image
PROP = [0.1, 0.5]  # Proportion of missing pixels (during training and testing)

# PnP hyperparameters
STEP = 1.0  # Step size of PnP
N_ITER = 100  # Number of PnP iterations

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

# Load test image
img = Image.open(IMG_PATH)
img = img.convert("L")
img_np = pil_to_np(img)
img = np_to_torch(img_np).type(DTYPE).to(DEVICE)

psnrs_rec = np.zeros((len(N_SAMPLES), len(PROP)))
psnrs_rec_ref = np.zeros((len(N_SAMPLES), len(PROP)))
psnrs_corrupted = np.zeros(len(PROP))
mse = nn.MSELoss()

for i_prop, prop in enumerate(PROP):

    # Corrupt the test image with a binary mask and noise
    mask = (
        torch.rand(img.shape, generator=RNG, dtype=DTYPE, device=DEVICE) > prop
    )
    noise = torch.randn(img.shape, generator=RNG, dtype=DTYPE, device=DEVICE)
    corrupted_img = img * mask + SIGMA_TEST_SAMPLE * noise
    corrupted_img_np = torch_to_np(corrupted_img)

    # Store psnr of the corrupted image
    psnr_corr = psnr(corrupted_img_np, img_np)
    psnrs_corrupted[i_prop] = psnr_corr

    for i_n, n_samples in enumerate(N_SAMPLES):

        print(f"prop = {prop:.2f}, n_samples = {n_samples}\n")

        denoiser_names = [CORRUPTED_DENOISER_NAME, CLEAN_DENOISER_NAME]
        file_names = [
            f"{name}_{n_samples}_{floor(prop * 100)}"
            for name in denoiser_names
        ]
        if TRAINING:

            # Create inpainted denoising dataset
            corrupted_dataset = DenoisingInpaintingDataset(
                path=CLEAN_DATA,
                n_samples=n_samples,
                noise_std=SIGMA_MAX_DENOISER,
                rng=RNG,
                prop=prop,
                sigma=SIGMA_TRAIN_SAMPLE,
                dtype=DTYPE,
                device=DEVICE,
            )

            # Create clean denoising dataset
            clean_dataset = DenoisingDataset(
                path=CLEAN_DATA,
                n_samples=n_samples,
                noise_std=SIGMA_MAX_DENOISER,
                rng=RNG,
                dtype=DTYPE,
                device=DEVICE,
            )

            datasets = [corrupted_dataset, clean_dataset]
            modes = ["denoising_inpainting", "denoising"]

            for dataset, file_name, mode in zip(datasets, file_names, modes):
                train_dataloader, val_dataloader = train_val_dataloaders(
                    dataset=dataset,
                    train_batch_size=BATCH_SIZE,
                    val_batch_size=VAL_BATCH_SIZE,
                    val_split_fraction=0.1,
                    np_rng=NP_RNG,
                    seed=SEED,
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
                    learning_rate=LR,
                    device=DEVICE,
                    n_epochs=N_EPOCHS,
                )
                print()

        for file_name, mode in zip(file_names, modes):
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
                for iter in range(1, N_ITER + 1):
                    grad = mask * (out - corrupted_img)
                    out -= STEP * grad
                    out = denoiser(out)
                    loss = mse(out * mask, corrupted_img)
                    if iter % 10 == 0:
                        print(f"Iteration {iter}, loss = {loss.item()}")

            out_np = torch_to_np(out)

            # Store PSNR
            psnr_rec = psnr(out_np, img_np)
            if mode == "denoising_inpainting":
                psnrs_rec[i_prop, i_n] = psnr_rec
            elif mode == "denoising":
                psnrs_rec_ref[i_prop, i_n] = psnr_rec

            print(f"prop = {prop:.2f}, n_samples = {n_samples}")
            print(f"psnr_rec = {psnr_rec:.2f}, mode={mode}")
            print(f"psnr_corr = {psnr_corr:.2f}")
            delta = time.time() - start_time
            delta = str(datetime.timedelta(seconds=delta))
            print(f"elapsed time: {delta}\n")

# Save the results
results_df = {
    "psnrs_rec": {"psnrs_rec": psnrs_rec},
    "psnrs_rec_ref": {"psnrs_rec_ref": psnrs_rec_ref},
    "psnrs_corrupted": {"psnrs_corrupted": psnrs_corrupted},
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("experiments/results/inpainting_samples_pnp.pickle")
