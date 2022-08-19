import datetime
from dl_inv_prob.common_utils import (
    pil_to_np,
)
from dl_inv_prob.dataset import (
    make_dataset,
)
from dl_inv_prob.dl import ConvolutionalInpainting
from dl_inv_prob.utils import psnr
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import time
import torch
import torch.optim

start_time = time.time()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Paths
DATA_PATH = "experiments/data"
CLEAN_DATA = os.path.join(DATA_PATH, "Train400")
IMG_PATH = os.path.join(DATA_PATH, "flowers.png")

# Hyperparameters
N_SAMPLES = 20 * np.arange(1, 6)  # Training samples
SIGMA_MAX_DENOISER = 0.3  # Max noise level to train the denoiser
SIGMA_TRAIN_SAMPLE = 0.0  # Noise level of "clean" training samples
SIGMA_TEST_SAMPLE = 0.1  # Noise level of the corrupted test image
PROP = 0.1 * np.arange(1, 6)  # Proportion of missing pixels
SIZE = 128  # Image size

# CDL hyperparameters
PROP_ATOM = 1 / 16  # Size of atoms with respect to image size
N_ATOMS = 10  # Number of atoms in the dictionary
LAMBD = 0.1  # Regularization parameter
DIM_ATOM = int(SIZE * PROP_ATOM)  # Atom size

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
img = Image.open(IMG_PATH).resize((SIZE, SIZE), Image.ANTIALIAS).convert("L")
img = pil_to_np(img)

psnrs_rec = np.zeros((len(N_SAMPLES), len(PROP)))
psnrs_rec_ref = np.zeros((len(N_SAMPLES), len(PROP)))
psnrs_corrupted = np.zeros(len(PROP))

# Fixed dictionary initialization
D_init = NP_RNG.normal(size=(N_ATOMS, 1, DIM_ATOM, DIM_ATOM))

for i_prop, prop in enumerate(PROP):

    # Corrupt the test image with a binary mask and noise
    mask = NP_RNG.binomial(1, 1 - prop, size=img.shape)
    noise = NP_RNG.standard_normal(img.shape)
    corrupted_img = img * mask + SIGMA_TEST_SAMPLE * noise

    # Store psnr of the corrupted image
    psnr_corr = psnr(corrupted_img, img)
    psnrs_corrupted[i_prop] = psnr_corr

    for i_n, n_samples in enumerate(N_SAMPLES):

        sample_paths = make_dataset(CLEAN_DATA, n_samples)

        # Create clean dataset
        clean_data = np.array(
            [
                pil_to_np(
                    Image.open(path)
                    .resize((SIZE, SIZE), Image.ANTIALIAS)
                    .convert("L")
                )[0, :, :]
                for path in sample_paths
            ]
        )

        # Create inpainted dataset
        corrupted_data = clean_data.copy()
        masks = NP_RNG.binomial(1, 1 - prop, size=corrupted_data.shape)
        noise = NP_RNG.standard_normal(corrupted_data.shape)
        corrupted_data = corrupted_data * masks + SIGMA_TRAIN_SAMPLE * noise

        datasets = [corrupted_data, clean_data]
        modes = ["corrupted", "clean"]

        for data, mode in zip(datasets, modes):
            # Learn a dictionary with CDL
            if mode == "corrupted":
                cdl = ConvolutionalInpainting(
                    n_components=N_ATOMS,
                    lambd=LAMBD,
                    init_D=D_init,
                    atom_height=DIM_ATOM,
                    atom_width=DIM_ATOM,
                    device=DEVICE,
                    rng=NP_RNG,
                )
                cdl.fit(data, masks)
            else:
                cdl = ConvolutionalInpainting(
                    n_components=N_ATOMS,
                    lambd=LAMBD,
                    init_D=D_init,
                    atom_height=DIM_ATOM,
                    atom_width=DIM_ATOM,
                    device=DEVICE,
                    rng=NP_RNG,
                )
                cdl.fit(data, np.ones(data.shape))

            # Reconstruct the image with the dictionary
            rec = cdl.rec(corrupted_img).squeeze()
            rec = np.clip(rec, 0, 1)

            # Store PSNR
            psnr_rec = psnr(rec, img)
            if mode == "corrupted":
                psnrs_rec[i_prop, i_n] = psnr_rec
            else:
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
    "n_samples": {"n_samples": N_SAMPLES},
    "props": {"props": PROP},
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("experiments/results/inpainting_samples_cdl.pickle")
