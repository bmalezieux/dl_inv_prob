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

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
TRAINING = True

# Paths

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(EXPERIMENTS, "data")
CLEAN_DATA = os.path.join(DATA_PATH, "Train400")
TEST_DATA = os.path.join(DATA_PATH, "Train400_test")
IMG_PATH = os.path.join(DATA_PATH, "flowers.png")
MODELS_PATH = os.path.join(EXPERIMENTS, "../dl_inv_prob/models/pretrained")
CORRUPTED_DENOISER_NAME = "DnCNN_denoising_inpainting"
CLEAN_DENOISER_NAME = "DnCNN_denoising"
RESULTS = os.path.join(EXPERIMENTS, "results/inpainting_pnp.csv")
SIZE = 180
N_TEST = 20

# Hyperparameters

hyperparams = {
    "sigma_max_denoiser": [0.05, 0.1, 0.3],
    "sigma_sample": [0, 0.02, 0.05, 0.1],
    "prop": np.arange(0.1, 1, 0.1),
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

    # # Corrupt the test image with a binary mask and noise
    # img, corrupted_img, mask = determinist_inpainting(
    #     img_path=IMG_PATH,
    #     prop=params["prop"],
    #     sigma=params["sigma_sample"],
    #     size=SIZE,
    #     seed=SEED,
    #     dtype=DTYPE,
    #     device=DEVICE,
    # )
    # mask = mask[None, :, :]
    # img = img[None, :, :]
    # corrupted_img = corrupted_img[None, :, :]
    # corrupted_img_np = torch_to_np(corrupted_img)
    # img_np = torch_to_np(img)

    # # ipdb.set_trace()

    # # Store psnr of the corrupted image
    # psnr_corr = psnr(corrupted_img_np, img_np)

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
    modes = ["denoising_inpainting", "denoising"]
    file_name = "DnCNN_pnp_inpainting"

    psnr_rec_supervised = 0
    psnr_rec_unsupervised = 0

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

        psnr_corr = 0

        for filename in tqdm(os.listdir(TEST_DATA)[:N_TEST]):
            img_test, corrupted_img_test, mask_test = determinist_inpainting(
                img_path=os.path.join(TEST_DATA, filename),
                prop=params["prop"],
                sigma=params["sigma_sample"],
                size=SIZE,
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
            if mode == "denoising_inpainting":
                psnr_rec_unsupervised += psnr_rec
            elif mode == "denoising":
                psnr_rec_supervised += psnr_rec
            psnr_corr += psnr(corrupted_img_test_np, img_test_np)

    psnr_rec_unsupervised /= N_TEST
    psnr_rec_supervised /= N_TEST
    psnr_corr /= N_TEST

    print(f"psnr_rec_unsupervised = {psnr_rec_unsupervised:.2f}")
    print(f"psnr_rec_supervised = {psnr_rec_supervised:.2f}")
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
