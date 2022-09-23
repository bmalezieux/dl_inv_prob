# %%
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# %%

img = Image.open("../data/flowers.png").convert("L")
y = np.array(img) / 255.
# %%

from dl_inv_prob.utils import gaussian_kernel
from scipy.signal import convolve

sigma = 0.3
kernel = gaussian_kernel(10, sigma)

size_x, size_y = y.shape

dirac_image = np.zeros((size_x, size_y))
dirac_image[size_x // 2, size_y // 2] = 1

result = convolve(dirac_image, kernel, mode="same")

f_dirac = np.fft.fft2(result)
mag_dirac = np.abs(f_dirac)

value = np.exp(-0.5 * 4) / np.sqrt(2 * np.pi * sigma ** 2)

ran_kernel = (mag_dirac > value)
ker_kernel = (mag_dirac < value)

# %%

f_img = np.fft.fft2(y)


img_ran = np.clip(np.abs(np.fft.ifft2(f_img * ran_kernel)), 0, 1)
img_ker = np.clip(np.abs(np.fft.ifft2(f_img * ker_kernel)), 0, 1)
# %%
img_ker.max()
# %%
plt.imshow(img_ker, cmap="gray")
# %%
plt.imshow(img_ran, cmap="gray")
# %%
img_ran.max()
# %%
