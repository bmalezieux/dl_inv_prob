import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_pickle('../results/deblurring_flower.pickle')
atoms = df.atoms.atoms
atoms_psd = df.atoms_psd.atoms_psd
kernel_psd = df.kernel_psd.kernel_psd
psnr = df.psnr.psnr

y_conv = np.load('../results/deblurring_flower_blurred.npy')
rec = np.load('../results/deblurring_flower_rec.npy')
y = np.load('../results/deblurring_flower_original.npy')

# Reconstruction
fig = plt.figure(figsize=(15, 10))

fig.add_subplot(131)
plt.imshow(y_conv, cmap='gray')
plt.title('Corrupted')

fig.add_subplot(132)
plt.imshow(rec, cmap='gray')
plt.title(f'Reconstruction (PSNR = {psnr:.2f})')

fig.add_subplot(133)
plt.imshow(y, cmap='gray')
plt.title('Original')

plt.savefig('../figures/deblurring_flower_rec.pdf')

# Atoms
fig = plt.figure(figsize=(15, 5))

for i, atom in enumerate(atoms):
    fig.add_subplot(2, 5, i + 1)
    atom = (atom - atom.min()) / (atom.max() - atom.min())
    plt.imshow(atom, cmap='gray')

plt.savefig('../figures/deblurring_flower_atoms.pdf')

# Atoms psd
fig = plt.figure(figsize=(15, 5))

for i, psd in enumerate(atoms_psd):
    fig.add_subplot(2, 5, i + 1)
    plt.imshow(psd, cmap='gray')

plt.savefig('../figures/deblurring_flower_atoms_psd.pdf')

# Kernel psd
fig = plt.figure(figsize=(15, 5))

plt.imshow(kernel_psd, cmap='gray')
plt.savefig('../figures/deblurring_flower_kernel_psd.pdf')
