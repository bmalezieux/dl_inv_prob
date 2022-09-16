# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

from dl_inv_prob.common_utils import (
    pil_to_np,
)
from dl_inv_prob.utils import psnr, is_divergence
from PIL import Image
from pathlib import Path

# %%
EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results")
DATA = os.path.join(EXPERIMENTS, "data")
IMG = os.path.join(DATA, "flowers.png")
SIZE = 50

img = Image.open(IMG).convert("L").resize((SIZE, SIZE),
                                          Image.ANTIALIAS)
img = pil_to_np(img)
y = np.array(img).squeeze()
# %%


def discrepancy_distance(y):  

    u = np.log(np.abs(np.fft.fft2(y)))
    u /= np.linalg.norm(u)
    s = 0
    for i, j, k, l in itertools.product(
        np.arange(u.shape[0]),
        np.arange(u.shape[1]),
        np.arange(u.shape[0]),
        np.arange(u.shape[1])):
        s += u[i, j] * u[k, l] * (np.log(np.abs(i - j) + np.abs(k - l)) + 1)
    return s
    
    
# %%
discrepancy_distance(y)
# %%
# %%
