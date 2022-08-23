from dl_inv_prob.dl import Deconvolution
from dl_inv_prob.utils import gaussian_kernel, psnr
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import fftshift, fft2
import torch
import torch.nn.functional as F

DEVICE = "cuda:3"
RNG = np.random.default_rng(2022)

dim_image = 200
dim_patch = 10
n_atoms = 10

img = Image.open('../data/flowers.png')
img_resize = img.resize((dim_image, dim_image), Image.ANTIALIAS).convert('L')
y = np.array(img_resize) / 255
y = torch.tensor(y, device=DEVICE, dtype=torch.float, requires_grad=False)
kernel = gaussian_kernel(10, 0.3)
kernel = torch.tensor(kernel, device=DEVICE, dtype=torch.float,
                      requires_grad=False)
y_conv = F.conv_transpose2d(y[None, None, :, :], kernel[None, None, :, :])

y_conv = y_conv.detach().cpu().numpy().squeeze()
kernel = kernel.detach().cpu().numpy().squeeze()
y = y.detach().cpu().numpy().squeeze()

dl = Deconvolution(
        n_atoms,
        init_D=None,
        device=DEVICE,
        rng=RNG,
        atom_height=dim_patch,
        atom_width=dim_patch,
        lambd=0.05
    )
dl.fit(y_conv[None, :, :], kernel)
D = dl.D_.squeeze()

rec = np.clip(dl.rec().squeeze(), 0, 1)
psnr_rec = psnr(rec, y)

atoms_psd = np.zeros((n_atoms, dim_patch, dim_patch))
for i, atom in enumerate(D):
    psd = np.log(1 + np.abs(fft2(atom)))
    psd = fftshift(psd)
    atoms_psd[i, :, :] = psd

kernel_psd = np.log(1 + np.abs(fft2(kernel)))
kernel_psd = fftshift(kernel_psd)

results_df = {
    "atoms": {"atoms": D},
    "atoms_psd": {"atoms_psd": atoms_psd},
    "kernel_psd": {"kernel_psd": kernel_psd},
    "psnr": {"psnr": psnr_rec}
}
results_df = pd.DataFrame(results_df)
results_df.to_pickle("../results/deblurring_flower.pickle")

np.save("../results/deblurring_flower_rec.npy", rec)
np.save("../results/deblurring_flower_blurred.npy", y_conv)
np.save("../results/deblurring_flower_original.npy", y)
