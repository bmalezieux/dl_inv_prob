from dl_inv_prob.common_utils import (
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
import random
import torch
import torch.optim
import torch.nn as nn
from pathlib import Path
import itertools
from tqdm import tqdm


DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Paths

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(EXPERIMENTS, "data")
CLEAN_DATA = os.path.join(DATA_PATH, "imagewoof_resized")
TEST_DATA = os.path.join(DATA_PATH, "imagewoof_resized_test")
RESULTS = os.path.join(EXPERIMENTS, "results/inpainting_pnp_number_samples_woof.csv")
N_TEST = 50

# Hyperparameters

hyperparams = {
    "sigma_max_denoiser": [0.05, 0.1, 0.3],
    "sigma_sample": [0, 0.02],
    "prop": [0.5],
    "n_samples": np.logspace(1, np.log(5000) / np.log(10), num=10, dtype=int),
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

mse = nn.MSELoss()
dict_results = {}

for params in tqdm(permuts_params):

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

    psnr_rec_supervised = 0
    psnr_rec_unsupervised = 0

    results = {
        "psnr_rec_unsupervised": psnr_rec_unsupervised,
        "psnr_rec_supervised": psnr_rec_supervised,
    }

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
        stop = False

        for i in range(params["n_epochs"] // 5):

            if stop:
                break

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

            for filename in os.listdir(TEST_DATA)[:N_TEST]:
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
                if mode == "denoising_inpainting":
                    psnr_rec_unsupervised += psnr_rec
                elif mode == "denoising":
                    psnr_rec_supervised += psnr_rec

            if mode == "denoising_inpainting":
                psnr_rec_unsupervised /= N_TEST
                if psnr_rec_unsupervised < results["psnr_rec_unsupervised"]:
                    stop = True
                else:
                    results["psnr_rec_unsupervised"] = psnr_rec_unsupervised
            elif mode == "denoising":
                psnr_rec_supervised /= N_TEST
                if psnr_rec_supervised < results["psnr_rec_supervised"]:
                    stop = True
                else:
                    results["psnr_rec_supervised"] = psnr_rec_supervised

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
