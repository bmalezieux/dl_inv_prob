import numpy as np

from copy import deepcopy
from pywt import wavedec2, waverec2


class SparseWavelets():
    """
    Inpainting with wavelet sparsity in 2D
    """
    def __init__(self, lambd, wavelet="db3"):

        self.wavelet = wavelet
        self.lambd = lambd

    def wavelet_transform(self, y):
        """ Returns the wavelet decomposition of an image y """
        z = wavedec2(y, self.wavelet, mode="zero")
        new_z = []
        for elt in z:
            new_z.append(np.array(elt))
        return new_z

    def wavelet_inv(self, z):
        """ Returns the wavelet reconstruction fropm coefficients z """
        new_z = []
        levels = len(z)
        for i in range(levels):
            new_z.append(tuple(z[i]))
        y = waverec2(new_z, self.wavelet)
        return y

    def st(self, z, reg):
        """ Soft thresholding """
        return np.sign(z) * np.maximum(0, np.abs(z) - reg)

    def compute_energy(self, res, z):
        """ Loss """
        norm1 = 0
        for elt in z:
            norm1 += np.sum(np.abs(elt))
        return 0.5 * np.sum(res ** 2) + self.lambd * norm1

    def rec(self, y, A, n_iter=1000, step=1.):
        """ Reconstruction process using (F)ISTA """
        t = 1.
        out = self.wavelet_transform(y)
        iterate = deepcopy(out)
        # E = []
        for i in range(n_iter):
            # Keep last iterate for FISTA
            iterate_old = deepcopy(iterate)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            coeff_fista = (t - 1.) / t_new

            # PGD
            y_wav = self.wavelet_inv(out)
            res = A * y_wav - y
            # E.append(self.compute_energy(res, out))
            out_wav = self.wavelet_transform(res)
            for k in range(len(out)):
                out[k] -= step * out_wav[k]
                out[k] = self.st(out[k], step * self.lambd)
                iterate[k] = out[k]
                out[k] = iterate[k]\
                    + coeff_fista * (iterate[k] - iterate_old[k])

            t = t_new

        return self.wavelet_inv(out)
