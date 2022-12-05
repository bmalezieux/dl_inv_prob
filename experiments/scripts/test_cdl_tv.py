#%%
from dl_inv_prob.utils import determinist_blurr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import random
import torch

from dl_inv_prob.common_utils import (
    torch_to_np,
)
from dl_inv_prob.dl import CDLBasic, ConvolutionalInpainting

EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = os.path.join(EXPERIMENTS, "data")
IMG = os.path.join(DATA, "flowers.png")
RESULTS = os.path.join(EXPERIMENTS, "results")
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
RESULT_FILE = "deblurring_tv_atoms.csv"

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

SIZE = 256

img = Image.open(IMG).convert("L").resize((SIZE, SIZE), Image.ANTIALIAS)
img = np.array(img, dtype=np.float32) / 255.0

#%%

lambd = 0.1
mu = 0.01
n_atoms = 20
dim_atoms = 10

# CDL without line search with fixes number of D step (max_iter_D)
cdl_basic = CDLBasic(
    n_components=n_atoms,
    lambd=lambd,
    atom_height=dim_atoms,
    atom_width=dim_atoms,
    max_iter_D=1,
    device=DEVICE,
    rng=NP_RNG,
    keep_dico=True,
)
cdl_basic.fit(img[None, :, :])
print(cdl_basic.path_loss)

# CDL with line search and only one D step
mask = np.ones((SIZE, SIZE), dtype=np.float32)
cdl = ConvolutionalInpainting(
    n_components=n_atoms,
    lambd=lambd,
    atom_height=dim_atoms,
    atom_width=dim_atoms,
    device=DEVICE,
    rng=NP_RNG,
    keep_dico=True,
)
cdl.fit(img[None, :, :], mask[None, :, :])
print(cdl.path_loss)
