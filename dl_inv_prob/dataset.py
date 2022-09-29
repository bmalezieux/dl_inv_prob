from .utils import determinist_inpainting, determinist_blurr
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F


def make_dataset(path, n_samples=None):
    """Return a list of image paths."""
    images = []
    for img in os.listdir(path):
        if n_samples is not None and len(images) >= n_samples:
            break
        if img.endswith(".png") or img.endswith(".JPEG") or img.endswith(".jpeg"):
            img_path = os.path.join(path, img)
            images.append(img_path)

    return images


class DenoisingDataset(Dataset):
    """Dataset to train a denoiser."""

    def __init__(
        self,
        path,
        noise_std,
        n_samples=None,
        rng=None,
        dtype=torch.float32,
        device="cpu",
    ):
        self.images = make_dataset(path, n_samples=n_samples)
        self.noise_std = noise_std
        self.rng = rng if rng is not None else torch.Generator(device=device)
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("L")
        clean = T.ToTensor()(img).to(self.device)
        noise = torch.randn(
            clean.shape,
            generator=self.rng,
            dtype=self.dtype,
            device=self.device,
        )
        random_sigma = (
            torch.rand(
                1, generator=self.rng, dtype=self.dtype, device=self.device
            )
            * self.noise_std
        )
        noisy = clean + random_sigma * noise

        return noisy, clean


class DenoisingInpaintingDataset(Dataset):
    def __init__(
        self,
        path,
        seed_inpainting=2022,
        n_samples=None,
        noise_std=0.3,
        rng=None,
        dtype=torch.float32,
        device="cpu",
        prop=0.1,
        sigma=0.0,
    ):
        self.images = make_dataset(path, n_samples)
        self.noise_std = noise_std
        self.rng = rng if rng is not None else torch.Generator(device=device)
        self.dtype = dtype
        self.device = device
        self.prop = prop
        self.sigma = sigma
        self.seed_inpaiting = seed_inpainting

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        original, corrupted, mask = determinist_inpainting(
            img_path=img_path,
            prop=self.prop,
            sigma=self.sigma,
            seed=self.seed_inpaiting,
            dtype=self.dtype,
            device=self.device,
        )
        noise = torch.randn(
            original.shape,
            generator=self.rng,
            dtype=self.dtype,
            device=self.device,
        )
        random_sigma = (
            torch.rand(
                1, generator=self.rng, dtype=self.dtype, device=self.device
            )
            * self.noise_std
        )
        noisy = mask * (corrupted + random_sigma * noise)

        return noisy, corrupted, mask


class DenoisingDeblurringDataset(Dataset):
    def __init__(
        self,
        path,
        n_samples=None,
        noise_std=0.3,
        rng=None,
        dtype=torch.float32,
        device="cpu",
        sigma=0.0,
        sigma_blurr=0.1,
        size_blurr=10
    ):
        self.images = make_dataset(path, n_samples)
        self.noise_std = noise_std
        self.rng = rng if rng is not None else torch.Generator(device=device)
        self.dtype = dtype
        self.device = device
        self.sigma = sigma
        self.sigma_blurr = sigma_blurr
        self.size_blurr = size_blurr

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        original, corrupted, blurr = determinist_blurr(
            img_path,
            self.sigma_blurr,
            self.size_blurr,
            self.sigma,
            dtype=self.dtype,
            device=self.device,
        )
        noise = torch.randn(
            corrupted.shape,
            generator=self.rng,
            dtype=self.dtype,
            device=self.device,
        )
        random_sigma = (
            torch.rand(
                1, generator=self.rng, dtype=self.dtype, device=self.device
            )
            * self.noise_std
        )
        noisy = corrupted + random_sigma * noise

        return noisy, corrupted, blurr


class SimpleDataset(Dataset):
    """Dataset from a directory of images."""

    def __init__(self, path, n_samples=None):
        self.images = make_dataset(path, n_samples)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")
        img = T.ToTensor()(img)

        return img


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_val_dataloaders(
    dataset,
    train_batch_size=128,
    val_batch_size=32,
    val_split_fraction=0.1,
    seed=None,
    np_rng=None,
    num_workers=0,
):
    """Return training and validation dataloaders."""
    indices = list(range(len(dataset)))
    split = int(val_split_fraction * len(dataset))

    if np_rng is None:
        np_rng = np.random.default_rng()
    val_idx = np_rng.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(val_idx))

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    train_sampler = SubsetRandomSampler(train_idx, generator=rng)
    val_sampler = SubsetRandomSampler(val_idx, generator=rng)

    train_dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        generator=rng,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    val_dataloader = DataLoader(
        dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        generator=rng,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    return train_dataloader, val_dataloader
