import numpy as np
import torch

from dl_inv_prob.dl import DictionaryLearning
from dl_inv_prob.utils import generate_data, recovery_score, generate_dico


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def test_same_path():
    rng = np.random.default_rng(100)
    dico = generate_dico(50, 50, rng=rng)
    Y, _ = generate_data(dico[None, :], 1000, rng=rng)
    Y = Y.transpose((2, 1, 0))

    masks = rng.binomial(1, 1, size=(1000, 50))
    A = np.concatenate([np.diag(mask)[None, :] for mask in masks])
    Y_inpainting = A @ Y

    D_init = generate_dico(100, 50, rng=rng)
    dl1 = DictionaryLearning(100, lambd=0.1, init_D=D_init, device=DEVICE,
                             rng=rng)
    dl1.fit(Y)

    dl2 = DictionaryLearning(100, lambd=0.1, init_D=D_init, device=DEVICE,
                             rng=rng)
    dl2.fit(Y_inpainting, A)

    assert recovery_score(dl1.D_, dl2.D_) == 1.0


def test_run_dl():
    rng = np.random.default_rng(100)
    dl = DictionaryLearning(50, device=DEVICE, rng=rng)
    A = rng.random(size=(10, 30, 50))
    Y = rng.random(size=(10, 30, 100))
    dl.fit(Y, A)
    assert dl.D_.shape == (50, 50)


def test_dl_denoising():
    rng = np.random.default_rng(100)
    dico = generate_dico(50, 50, rng=rng)
    Y, _ = generate_data(dico[None, :], 1000, rng=rng)
    dl = DictionaryLearning(50, lambd=0.1, device=DEVICE, rng=rng)
    dl.fit(Y)
    assert recovery_score(dl.D_, dico) >= 0.99


def test_dl_inv_prob():
    rng = np.random.default_rng(100)
    dico = generate_dico(50, 50, rng=rng)
    A = rng.normal(size=(1, 50, 50))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    Y, _ = generate_data(A @ dico, 1000, rng=rng)
    dl = DictionaryLearning(50, lambd=0.1, device=DEVICE, rng=rng)
    dl.fit(Y, A)
    assert recovery_score(dl.D_, dico) >= 0.9
