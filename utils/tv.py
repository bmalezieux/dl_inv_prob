import numpy as np


class ProxTV:
    """
    Inpainting with Total Variation in 2D
    """

    def __init__(self, lambd=1, n_iter=1000):
        self.lambd = lambd
        self.n_iter = n_iter

    def grad(self, im):
        im_copy = im.astype(float)
        fx = np.zeros(im_copy.shape)
        fy = np.zeros(im_copy.shape)
        fx[:-1, :] = im_copy[1:, :] - im_copy[:-1, :]
        fy[:, :-1] = im_copy[:, 1:] - im_copy[:, :-1]
        return [fx, fy]

    def div(self, g):
        gx, gy = g[0], g[1]
        divx = np.zeros(g[0].shape)
        divy = np.zeros(g[1].shape)
        divx[1:, :] = gx[1:, :] - gx[:-1, :]
        divy[:, 1:] = gy[:, 1:] - gy[:, :-1]
        divx[0, :] = gx[0, :]
        divx[-1, :] = -gx[-2, :]
        divy[:, 0] = gy[:, 0]
        divy[:, -1] = -gy[:, -2]
        return divx + divy

    def proxG(self, p):
        return p / np.maximum(1, np.abs(p) / self.lambd)

    def fit(self, y, A=None):
        self.y = y[0, :, :]
        if A is not None:
            self.A = A[0, :, :]
        else:
            self.A = (y != 0).astype(float)
        tau = 0.1
        sigma = 1 / (8 * self.lambd * self.lambd * tau)
        u = self.y.copy()
        px = np.zeros(self.y.shape)
        py = np.zeros(self.y.shape)
        for _ in range(self.n_iter):
            uold = u.copy()
            u = (
                u
                - tau * (self.A * u - self.y)
                + tau * self.lambd * self.div([px, py])
            )
            gu = self.grad(2 * u - uold)
            px = self.proxG(px + sigma * gu[0])
            py = self.proxG(py + sigma * gu[1])

        self.res = u

    def rec(self):
        return self.res
