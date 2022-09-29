import numpy as np

from copy import deepcopy
from pywt import wavedec2, waverec2
from scipy.signal import convolve, correlate
from scipy.fft import fft2


class SparseWavelets:
    """
    Inpainting with wavelet sparsity in 2D
    """

    def __init__(self, lambd, wavelet="db3", n_iter=1000, step=1.0):

        self.wavelet = wavelet
        self.lambd = lambd
        self.n_iter = n_iter
        self.step = step

    def wavelet_transform(self, y):
        """Returns the wavelet decomposition of an image y"""
        z = wavedec2(y, self.wavelet, mode="zero")
        new_z = []
        for elt in z:
            new_z.append(np.array(elt))
        return new_z

    def wavelet_inv(self, z):
        """Returns the wavelet reconstruction fropm coefficients z"""
        new_z = []
        levels = len(z)
        for i in range(levels):
            new_z.append(tuple(z[i]))
        y = waverec2(new_z, self.wavelet)
        return y

    def st(self, z, reg):
        """Soft thresholding"""
        return np.sign(z) * np.maximum(0, np.abs(z) - reg)

    def compute_energy(self, res, z):
        """Loss"""
        norm1 = 0
        for elt in z:
            norm1 += np.sum(np.abs(elt))
        return 0.5 * np.sum(res**2) + self.lambd * norm1

    def fit(self, y, A):
        """Reconstruction process using (F)ISTA"""
        y = y.squeeze()
        A = A.squeeze()
        t = 1.0
        out = self.wavelet_transform(y)
        iterate = deepcopy(out)
        # E = []
        for i in range(self.n_iter):
            # Keep last iterate for FISTA
            iterate_old = deepcopy(iterate)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            coeff_fista = (t - 1.0) / t_new

            # PGD
            y_wav = self.wavelet_inv(out)
            res = A * y_wav - y
            # E.append(self.compute_energy(res, out))
            out_wav = self.wavelet_transform(res)
            for k in range(len(out)):
                out[k] -= self.step * out_wav[k]
                out[k] = self.st(out[k], self.step * self.lambd)
                iterate[k] = out[k]
                out[k] = iterate[k] + coeff_fista * (
                    iterate[k] - iterate_old[k]
                )

            t = t_new

        self.wavelet_res = out

    def rec(self):
        return self.wavelet_inv(self.wavelet_res)


class SparseWaveletsDeblurring(SparseWavelets):
    """
    Deconvolution with wavelet sparsity in 2D
    """
    def __init__(self, lambd=1., wavelet="db3", n_iter=1000, step=1.0):
        super().__init__(lambd, wavelet, n_iter)

    def fit(self, y, A):
        """Reconstruction process using (F)ISTA"""
        self.y = y.squeeze()
        self.A = A.squeeze()

        fourier_A = fft2(self.A)
        L = np.max(np.real(fourier_A * np.conj(fourier_A)))
        self.step /= L

        t = 1.0
        out = self.wavelet_transform(correlate(self.y, self.A, mode="valid"))
        iterate = deepcopy(out)
        # E = []
        for i in range(self.n_iter):
            # Keep last iterate for FISTA
            iterate_old = deepcopy(iterate)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            coeff_fista = (t - 1.0) / t_new

            # PGD
            y_wav = self.wavelet_inv(out)
            res = convolve(y_wav, self.A, mode="full") - self.y
            # E.append(self.compute_energy(res, out))
            out_wav = self.wavelet_transform(
                correlate(res, self.A, mode="valid")
            )
            for k in range(len(out)):
                out[k] -= self.step * out_wav[k]
                out[k] = self.st(out[k], self.step * self.lambd)
                iterate[k] = out[k]
                out[k] = iterate[k] + coeff_fista * (
                    iterate[k] - iterate_old[k]
                )

            t = t_new

        self.wavelet_res = out
