import jsonlines
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Memory
from submitit import AutoExecutor
from submitit.helpers import as_completed

from dl_inv_prob.dl import Inpainting
from dl_inv_prob.utils import (create_patches_overlap, generate_dico,
                               patch_average, psnr, recovery_score)


# Paths and constants
MAIN_PATH = Path(__file__).parent.parent
CACHE_PATH = MAIN_PATH / "__cache__"
DATA_PATH = MAIN_PATH / 'data' / 'flowers.png'
RESULTS_PATH = MAIN_PATH / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
JSONL_PATH = RESULTS_PATH / "inpainting_patches.jsonl"

mem = Memory(CACHE_PATH, verbose=0)

SEED = 2022
N_EXP = 20
N_ATOMS = 100
N_ITERS = 100

dims_image = [128, 256, 502]
dim_patch = 10
patch_len = dim_patch ** 2

s_values = np.arange(0.1, 1.0, 0.1)


@mem.cache
def run_one(s_values, img, dim_image, dim_patch, n_exp, seed):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    img = np.array(img.resize((dim_image, dim_image))) / 255
    y, _ = create_patches_overlap(img, dim_patch)
    trivial_masks = np.ones(y.shape)

    D_init = generate_dico(N_ATOMS, patch_len, rng=rng)
    dl = Inpainting(
        N_ATOMS, init_D=D_init, device=device, rng=rng, max_iter=N_ITERS
    )
    dl.fit(y[:, :, None], trivial_masks[:, :, None])
    D_no_inpainting = dl.D_

    res = []
    for s in tqdm(s_values, desc="Sparsity"):
        A = rng.binomial(1, 1 - s, size=(dim_image, dim_image))
        img_inpainting = img * A
        y_inpainting, masks = create_patches_overlap(img_inpainting, dim_patch, A)

        dl_inpainting = Inpainting(
            N_ATOMS, init_D=D_init, device=device, rng=rng, max_iter=N_ITERS
        )
        dl_inpainting.fit(y_inpainting[:, :, None], masks[:, :, None])
        D_inpainting = dl_inpainting.D_

        with torch.no_grad():
            codes = dl_inpainting(dl_inpainting.Y_tensor).detach().cpu().numpy()
        weights = np.abs(codes).sum(axis=(0, 2))

        rec_patches = dl_inpainting.rec(y_inpainting[:, :, None])
        rec = patch_average(rec_patches, dim_patch, dim_image, dim_image)
        rec = np.clip(rec, 0, 1)

        result = {
            "dim_image": dim_image,
            "n_exp": n_exp,
            "s": float(s),
            "score": float(recovery_score(D_inpainting, D_no_inpainting)),
            "score_weighted": float(recovery_score(D_inpainting, D_no_inpainting, weights)),
            "psnr": float(psnr(rec, img)),
            "psnr_corrupted": float(psnr(img_inpainting, img))
        }
        res.append(result)
    print(f"Results for dim_image={dim_image}, seed={seed}: {res}")
    return res


# Setup the parallel executor
executor = AutoExecutor(MAIN_PATH / "submitit_logs")
executor.update_parameters(
    slurm_time="03:00:00",
    slurm_gres="gpu:1",
    slurm_cpus_per_task=8,
    slurm_partition="gpu_p2",
    slurm_account="lsd@v100",
    # slurm_constraint="v100-32g",
    slurm_setup=[
        "module purge",
        "module load pytorch-gpu"
    ]
)

# Main processing
rng = np.random.default_rng(SEED)
img_orig = Image.open(DATA_PATH).convert('L')

futures = []
with executor.batch():
    for n_exp in range(N_EXP):
        for dim_image in dims_image:

            futures.append(
                executor.submit(
                    run_one,
                    s_values,
                    img_orig,
                    dim_image, dim_patch,
                    n_exp=n_exp,
                    seed=int(rng.integers(0, 2**31-1)))
            )

with jsonlines.open(JSONL_PATH, mode='w') as writer:
    for f in tqdm(as_completed(futures), total=len(futures), desc="Collecting Results"):
        [writer.write(result) for result in f.result()]
