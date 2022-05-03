import numpy as np

from dl_inv_prob.dl import DictionaryLearning
from dl_inv_prob.utils import generate_data, recovery_score, generate_dico


RNG = np.random.default_rng(100)
DEVICE = "cuda:1"


def test_run_dl():
    dl = DictionaryLearning(50, device=DEVICE, rng=RNG)
    A = RNG.random(size=(10, 30, 50))
    Y = RNG.random(size=(10, 30, 100))
    dl.fit(Y, A)
    assert dl.D_.shape == (50, 50)


def test_dl_denoising():
    dico = generate_dico(50, 50, rng=RNG)
    Y, _ = generate_data(dico[None, :], 1000, rng=RNG)
    dl = DictionaryLearning(50, lambd=0.1, device=DEVICE, rng=RNG)
    dl.fit(Y)
    assert recovery_score(dl.D_, dico) >= 0.99


def test_dl_inv_prob():
    dico = generate_dico(50, 50, rng=RNG)
    A = RNG.random(size=(1, 50, 50))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    Y, _ = generate_data(A @ dico, 1000, rng=RNG)
    dl = DictionaryLearning(50, lambd=0.1, device=DEVICE, rng=RNG)
    dl.fit(Y, A)
    assert recovery_score(dl.D_, dico) >= 0.9
