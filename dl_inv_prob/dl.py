import numpy as np
from scipy.signal.windows import tukey
import time
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


class DictionaryLearning(nn.Module):
    """
    Dictionary learning with alternating minimization

    Parameters
    ----------
    n_components : int
        Number of atoms in the dictionary.
    n_iter : int
        Number of unrolled iterations.
    lambd : float
        Regularization parameter.
        Default : 0.1.
    init_D : np.array, shape (width, height)
        Initialization for the dictionary.
        If None, the dictionary is initializaed from the data.
    device : str
        Device where the code is run ["cuda", "cpu"].
        If None, "cuda" is chosen if available.
    keep_dico : bool
        If True, keeps dictionaries, losses and times
        during optimization.
    rng : np.random.Generator
    tol : float
        Tolerance for outer problem

    Attributes
    ----------
    device : str
        Device where the code is run ["cuda", "cpu"]
    lambd : float
        Regularization parameter
    n_iter : int
        Number of unrolled iterations
    n_components : int
        Number of atoms in the dictionary
    dim_x : int
        Number of atoms
    dim_y : int
        Dimension of the measurements
    dim_signal : int
        Dimension of the signal
    init_D : np.array, shape (dim_signal, dim_x)
        Initialization for the dictionary
    lipschitz : float
        Lipschitz constant of the current dictionary
    operator : torch.Tensor, shape (n_matrices, dim_y, dim_signal)
        Measurement matrices
    D : torch.Tensor, shape (dim_signal, dim_x)
        Current dictionary
    steps : torch.Tensor, shape (n_iter)
        steps sizes for the sparse coding algorithm
    Y_tensor : torch.Tensor, shape (n_matrices, dim_y, number of data)
        data
    """

    def __init__(
        self,
        n_components,
        lambd=0.1,
        max_iter=100,
        init_D=None,
        step=1.0,
        algo="fista",
        keep_dico=False,
        tol=1e-6,
        rng=None,
        device=None,
    ):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Regularization parameter
        self.lambd = lambd

        # Algorithm parameters
        self.max_iter = max_iter
        self.step = step
        self.algo = algo
        self.tol = tol

        # Number of atoms
        self.n_components = n_components

        # Shape
        self.dim_x = n_components
        self.dim_y = None
        self.dim_signal = None
        self.n_matrices = None

        # Dictionary
        self.init_D = init_D
        self.lipschitz = None

        # Parameters for experiments
        self.keep_dico = keep_dico
        self.rng = rng

        # Tensors
        self.D = None
        self.Y_tensor = None
        self.operator = None

    @property
    def D_(self):
        """Returns the current dictionary
        np.array (dim_signal, n_components)"""
        return self.D.detach().to("cpu").numpy()

    def rescale(self, atoms="columns"):
        """
        Constrains the dictionary to have normalized atoms

        Returns
        -------
        norm_atoms : torch.Tensor, shape (dim_x)
            Contains the norms of the current atoms.
        """
        with torch.no_grad():
            norm_atoms = torch.norm(self.D, dim=0)
            norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1.0
            self.D /= norm_atoms
        return norm_atoms

    def unscale(self, norm_atoms):
        """
        Cancels the scaling using norms previously computed

        Parameters
        ----------
        norm_atoms : torch.Tensor, shape (dim_x)
            Contains the norms of the current atoms.
            Computed by rescale()
        """
        with torch.no_grad():
            self.D *= norm_atoms

    def compute_lipschitz(self):
        """Computes an upper bound of the
        Lipschitz constant of the dictionary for each matrix"""
        with torch.no_grad():
            product = torch.matmul(self.operator, self.D)
            self.lipschitz = (
                torch.norm(
                    torch.matmul(product.transpose(1, 2), product), dim=(1, 2)
                )
                .max()
                .item()
            )
            if self.lipschitz == 0:
                self.lipschitz = 1.0

    def forward(self, y):
        """
        (F)ISTA-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (n_matrices, dim_y, number of data)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape (n_matrices, dim_x, number of data)
            Approximation of the sparse code associated to y
        """
        out = torch.zeros(
            (self.n_matrices, self.dim_x, y.shape[2]),
            dtype=torch.float,
            device=self.device,
        )

        # For FISTA
        t = 1.0
        iterate = out.clone()

        # Computing step and product
        step = 1.0 / self.lipschitz
        product = torch.matmul(self.operator, self.D)

        for i in range(self.max_iter):
            # Keep last iterate for FISTA
            iterate_old = iterate.clone()

            # Gradient descent
            gradient = torch.matmul(
                product.transpose(1, 2), torch.matmul(product, out) - y
            )
            out = out - step * gradient

            # Thresholding
            thresh = torch.abs(out) - step * self.lambd
            out = torch.sign(out) * F.relu(thresh)

            iterate = out.clone()
            # Apply momentum for FISTA
            if self.algo == "fista":
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
                out = iterate + ((t - 1.0) / t_new) * (iterate - iterate_old)
                t = t_new

        return out

    def cost(self, y, x):
        """Cost function"""
        product = torch.matmul(self.operator, self.D)

        res = torch.matmul(product, x) - y
        l2 = (res * res).sum()
        l1 = torch.abs(x).sum()

        return 0.5 * l2 + self.lambd * l1

    def line_search(self, step, loss):
        """
        Gradient descent step with line search

        Parameters
        ----------
        step : float
            Starting step for line search.
        loss : float
            Current value of the loss.

        Returns
        -------
        future_step : float
            Value of the future starting step size.
        end : bool
            True if the optimization is done, False otherwise.
        """
        # Line search parameters
        beta = 0.5

        # Learning rate
        t = step

        ok = False
        end = False
        norm_atoms = None

        with torch.no_grad():
            # Learning step
            self.D -= beta * t * self.D.grad

            init = True

            while not ok:
                if not init:
                    # Unscaling
                    self.unscale(norm_atoms)
                    # Backtracking
                    self.D -= (beta - 1) * t * self.D.grad
                else:
                    init = False

                # Rescaling
                norm_atoms = self.rescale()

                # Computing step
                self.compute_lipschitz()

                # Computing loss with new parameters
                current_cost = self.cost(
                    self.Y_tensor, self.forward(self.Y_tensor)
                ).item()

                if current_cost < loss:
                    ok = True
                else:
                    t *= beta

                if t < 1e-20:
                    # Unscaling
                    self.unscale(norm_atoms)
                    # Stopping criterion
                    self.D += t * self.D.grad
                    # Rescaling
                    self.rescale()

                    # Computing step
                    self.compute_lipschitz()

                    ok = True
                    end = True

        # Avoiding numerical instabitility in the step size
        future_step = min(10 * t, 1e4)

        # future_step = t
        return future_step, end

    def training_process(self, backprop=True):
        """
        Training function, with backtracking line search.

        Returns
        -------
        float
            Final value of the loss after training.
        """
        # Initial backtracking step
        step = self.step
        end = False

        old_loss = None
        self.path_optim = []
        self.path_loss = []
        self.path_times = [0]
        start = time.time()

        while not end:

            # Computing loss
            with torch.no_grad():
                out = self.forward(self.Y_tensor)
            loss = self.cost(self.Y_tensor, out)

            # Keep track of the dictionaries
            if self.keep_dico:
                self.path_optim.append(self.D_)
                self.path_loss.append(loss.item())

            # Computing gradients
            loss.backward()

            # Gradient correction
            if self.cov_inv is not None:
                with torch.no_grad():
                    self.D.grad.data = torch.matmul(
                        self.cov_inv, self.D.grad.data
                    )

            # Line search
            step, end = self.line_search(step, loss)

            # Checking termination
            if old_loss is not None:
                if np.abs(old_loss - loss.item()) / old_loss < self.tol:
                    end = True

            old_loss = loss.item()

            # Putting the gradients to zero
            self.D.grad.zero_()

            if self.keep_dico:
                self.path_times.append(time.time() - start)

        if self.keep_dico:
            self.path_optim.append(self.D_)

        return loss.item()

    def fit(self, Y, A=None, cov_inv=None):
        """
        Training procedure

        Parameters
        ----------
        Y : np.array, shape (n_matrices, dim_y, data_size)
            Observations to be processed.
        A : np.array, shape (n_matrices, dim_y, dim_signal)
            Measurement matrices.
            If set to None, A is considered to be the identity.
            Default : None.
        cov_inv : np.array, shape (dim_signal, dim_signal)
            Inverse of covariance A.T @ A

        Returns
        -------
        loss : float
            Final value of the loss after training.
        """

        # Dimension
        self.dim_y = Y.shape[1]

        # Operator
        if A is None:
            self.dim_signal = self.dim_y
            self.operator = torch.eye(
                self.dim_y, device=self.device, dtype=torch.float
            )[None, :]
        else:
            self.dim_signal = A.shape[2]
            self.operator = torch.from_numpy(A).float().to(self.device)

        self.n_matrices = self.operator.shape[0]

        # Covariance
        if cov_inv is None:
            self.cov_inv = None
        else:
            self.cov_inv = torch.from_numpy(cov_inv).float().to(self.device)

        # Dictionary
        if self.init_D is None:
            if self.rng is None:
                self.rng = np.random.get_default_rng()
            choice = self.rng.choice(Y.shape[2], self.n_components)
            dico = self.rng.normal(size=(self.dim_signal, self.n_components))
            if A is None:
                dico = Y[0, :, choice]
            self.D = nn.Parameter(
                torch.tensor(dico, device=self.device, dtype=torch.float)
            )
        else:
            dico_tensor = torch.from_numpy(self.init_D).float().to(self.device)
            self.D = nn.Parameter(dico_tensor)

        # Scaling and computing lipschitz
        self.rescale()
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(Y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss

    def rec(self, Y):
        """
        Reconstruct the signal with one forward pass from the data.

        Parameters
        ----------
        Y : np.array, shape (n_matrices, dim_y, data_size)
            Observations to be processed.

        Returns
        -------
        rec : np.array, shape (n_matrices, dim_y, data_size)
            Reconstructed signal
        """
        with torch.no_grad():
            Y_tensor = torch.from_numpy(Y).float().to(self.device)
            x = self.forward(Y_tensor)
            rec = torch.matmul(self.D, x)

        return rec.detach().to("cpu").numpy()


class Inpainting(DictionaryLearning):
    def __init__(
        self,
        n_components,
        lambd=0.1,
        max_iter=100,
        init_D=None,
        step=1.0,
        algo="fista",
        keep_dico=False,
        tol=1e-6,
        rng=None,
        device=None,
    ):
        super().__init__(
            n_components=n_components,
            lambd=lambd,
            max_iter=max_iter,
            init_D=init_D,
            step=step,
            algo=algo,
            keep_dico=keep_dico,
            tol=tol,
            rng=rng,
            device=device,
        )

        self.masks = None

    def compute_lipschitz(self):
        """
        Computes an upper bound of the Lipschitz constant of the dictionary.
        """
        with torch.no_grad():
            product = self.masks * self.D
            self.lipschitz = (
                torch.norm(
                    torch.matmul(product.transpose(1, 2), product), dim=(1, 2)
                )
                .max()
                .item()
            )
            if self.lipschitz == 0:
                self.lipschitz = 1.0

    def cost(self, y, x):
        """
        Compute the LASSO-like cost.

        Parameters
        ----------
        y : torch.Tensor, shape (n_matrices, dim_y, data_size)
            Input signal
        x : torch.Tensor, shape (n_matrices, dim_x, data_size)
            Sparse codes

        Returns
        -------
        cost : float
            Cost value
        """
        product = self.masks * self.D

        res = torch.matmul(product, x) - y
        l2 = (res * res).sum()
        l1 = torch.abs(x).sum()

        return 0.5 * l2 + self.lambd * l1

    def forward(self, y):
        """
        (F)ISTA-like forward pass.

        Parameters
        ----------
        y : torch.Tensor, shape (n_matrices, dim_y, data_size)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape (n_matrices, dim_x, data_size)
            Approximation of the sparse code associated to y
        """
        out = torch.zeros(
            (self.n_matrices, self.dim_x, y.shape[2]),
            dtype=torch.float,
            device=self.device,
        )

        # For FISTA
        t = 1.0
        iterate = out.clone()

        # Computing step and product
        step = 1.0 / self.lipschitz
        product = self.masks * self.D

        for i in range(self.max_iter):
            # Keep last iterate for FISTA
            iterate_old = iterate.clone()

            # Gradient descent
            gradient = torch.matmul(
                product.transpose(1, 2), torch.matmul(product, out) - y
            )
            out = out - step * gradient

            # Thresholding
            thresh = torch.abs(out) - step * self.lambd
            out = torch.sign(out) * F.relu(thresh)

            iterate = out.clone()
            # Apply momentum for FISTA
            if self.algo == "fista":
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
                out = iterate + ((t - 1.0) / t_new) * (iterate - iterate_old)
                t = t_new

        return out

    def fit(self, Y, masks, cov_inv=None):
        """
        Training procedure.

        Parameters
        ----------
        Y : ndarray, shape (n_matrices, dim_y, data_size)
            Observations to be processed
        masks : ndarray, shape (n_matrices, dim_y)
            Diagonal masks
        cov_inv : ndarray, shape (dim_signal, dim_signal)
            Inverse of covariance A.T @ A

        Returns
        -------
        loss : float
            Final value of the loss after training
        """
        # Dimension
        self.dim_y = Y.shape[1]
        self.dim_signal = Y.shape[1]

        # masks
        self.masks = torch.from_numpy(masks).float().to(self.device)
        self.n_matrices = self.masks.shape[0]

        # Covariance
        if cov_inv is None:
            self.cov_inv = torch.eye(
                self.dim_signal, device=self.device, dtype=torch.float
            )
        else:
            self.cov_inv = torch.from_numpy(cov_inv).float().to(self.device)

        # Dictionary
        if self.init_D is None:
            if self.rng is None:
                self.rng = np.random.get_default_rng()
            dico = self.rng.normal(size=(self.dim_signal, self.n_components))
            self.D = nn.Parameter(
                torch.tensor(dico, device=self.device, dtype=torch.float)
            )
        else:
            dico_tensor = torch.from_numpy(self.init_D).float().to(self.device)
            self.D = nn.Parameter(dico_tensor)

        # Scaling and computing lipschitz
        self.rescale()
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(Y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss


class ConvolutionalInpainting(DictionaryLearning):
    def __init__(
        self,
        n_components,
        atom_height,
        atom_width,
        lambd=0.1,
        max_iter=100,
        init_D=None,
        step=1.0,
        algo="fista",
        alpha=None,
        keep_dico=False,
        tol=1e-6,
        rng=None,
        device=None,
    ):
        super().__init__(
            n_components=n_components,
            lambd=lambd,
            max_iter=max_iter,
            init_D=init_D,
            step=step,
            algo=algo,
            keep_dico=keep_dico,
            tol=tol,
            rng=rng,
            device=device,
        )

        self.masks = None
        self.conv = F.conv2d
        self.convt = F.conv_transpose2d
        self.atom_height = atom_height
        self.atom_width = atom_width

        self.alpha = alpha
        if alpha is not None:
            tukey_x = tukey(atom_height, alpha)
            tukey_y = tukey(atom_width, alpha)
            window = tukey_x[None, :] * tukey_y[:, None]
            self.window = torch.tensor(
                window,
                device=self.device,
                dtype=torch.float,
                requires_grad=False,
            )[None, None, :, :]

    def rescale(self):
        """
        Constrains the dictionary to have normalized atoms.

        Returns
        -------
        norm_atoms : torch.Tensor, shape (n_atoms, 1)
            Previous norms of the atoms
        """
        with torch.no_grad():
            norm_atoms = torch.linalg.norm(self.D, dim=(2, 3), keepdim=True)
            norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1.0
            self.D /= norm_atoms
        return norm_atoms

    def unscale(self, norm_atoms):
        """
        Cancels the scaling using norms previously computed.

        Parameters
        ----------
        norm_atoms : torch.Tensor, shape (n_atoms, 1)
            Contains the norms of the current atoms computed by rescale()
        """
        with torch.no_grad():
            self.D *= norm_atoms

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        with torch.no_grad():
            fourier_prior = fft.fftn(self.D, axis=(2, 3))
            self.lipschitz = (
                torch.max(
                    torch.max(
                        torch.real(fourier_prior * torch.conj(fourier_prior)),
                        dim=3,
                    )[0],
                    dim=2,
                )[0]
                .sum()
                .item()
            )
            if self.lipschitz == 0:
                self.lipschitz = 1.0

    def cost(self, y, x):
        """
        Compute the LASSO-like convolutional cost.

        Parameters
        ----------
        y : torch.Tensor, shape (batch_size, im_height, im_width)
            Input signal
        x : torch.Tensor,
            shape (batch_size, n_atoms, im_height - atom_height + 1,
                                        im_width - atom_width + 1)
            Sparse codes

        Returns
        -------
        cost : float
            Cost value
        """
        if self.alpha is None:
            dico = self.D
        else:
            dico = self.D * self.window
        rec = self.convt(x, dico)
        diff = self.masks * rec - y
        l2 = diff.ravel() @ diff.ravel()
        l1 = torch.sum(torch.abs(x))
        cost = 0.5 * l2 + self.lambd * l1

        return cost

    def forward(self, y):
        """
        (F)ISTA-like forward pass.

        Parameters
        ----------
        y : torch.Tensor, shape (batch_size, im_height, im_weight)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape (batch_size, 1, im_height - atom_height + 1,
                                                  im_width - atom_width + 1)
            Approximation of the sparse code associated to y
        """
        batch_size, im_height, im_width = y.shape
        out = torch.zeros(
            (
                batch_size,
                self.n_components,
                im_height - self.atom_height + 1,
                im_width - self.atom_width + 1,
            ),
            dtype=torch.float,
            device=self.device,
        )

        # For FISTA
        t = 1.0
        iterate = out.clone()

        # Computing step and product
        step = 1.0 / self.lipschitz

        for i in range(self.max_iter):
            if self.alpha is None:
                dico = self.D
            else:
                dico = self.D * self.window
            # Keep last iterate for FISTA
            iterate_old = iterate.clone()

            # Gradient descent
            rec = self.convt(out, dico)
            diff = self.masks[:, None, :, :] * (rec - y[:, None, :, :])
            # print(
            #     rec.shape,
            #     y.shape,
            #     self.masks.shape,
            #     diff.shape,
            #     dico.shape,
            #     out.shape,
            # )
            gradient = self.conv(diff, dico)
            out = out - step * gradient

            # Thresholding
            thresh = torch.abs(out) - step * self.lambd
            out = F.relu(thresh)

            iterate = out.clone()
            # Apply momentum for FISTA
            if self.algo == "fista":
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
                out = iterate + ((t - 1.0) / t_new) * (iterate - iterate_old)
                t = t_new

        return out

    def training_process(self):
        """
        Training function, with backtracking line search.

        Returns
        -------
        loss : float
            Final value of the loss after training
        """
        # Initial backtracking step
        step = self.step
        end = False

        old_loss = None
        self.path_optim = []
        self.path_loss = []
        self.path_times = [0]
        start = time.time()

        while not end:

            # Computing loss
            with torch.no_grad():
                out = self.forward(self.Y_tensor)
            loss = self.cost(self.Y_tensor, out)

            # Keep track of the dictionaries
            if self.keep_dico:
                self.path_optim.append(self.D_)
                self.path_loss.append(loss.item())

            # Computing gradients
            loss.backward()

            # Line search
            step, end = self.line_search(step, loss)

            # Checking termination
            if old_loss is not None:
                if np.abs(old_loss - loss.item()) / old_loss < self.tol:
                    end = True

            old_loss = loss.item()

            # Putting the gradients to zero
            self.D.grad.zero_()

            if self.keep_dico:
                self.path_times.append(time.time() - start)

        if self.keep_dico:
            self.path_optim.append(self.D_)

        return loss.item()

    def fit(self, Y, masks):
        """
        Training procedure.

        Parameters
        ----------
        Y : ndarray, shape (batch_size, im_height, im_weight)
            Observations to be processed
        masks : ndarray, shape (batch_size, im_height, im_width)
            Image masks

        Returns
        -------
        loss : float
            Final value of the loss after training
        """
        # masks
        self.masks = torch.from_numpy(masks).float().to(self.device)

        # Dictionary
        if self.init_D is None:
            if self.rng is None:
                self.rng = np.random.get_default_rng()
            dico = self.rng.normal(
                size=(self.n_components, 1, self.atom_height, self.atom_width)
            )
            self.D = nn.Parameter(
                torch.tensor(dico, device=self.device, dtype=torch.float)
            )
        else:
            dico_tensor = torch.from_numpy(self.init_D).float().to(self.device)
            self.D = nn.Parameter(dico_tensor)

        # Scaling and computing lipschitz
        self.rescale()
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(Y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss

    def rec(self, Y=None):
        """
        Reconstruct the image with the learned atoms and sparse codes.

        Parameters
        ----------
        Y : torch.Tensor, shape (batch_size, im_height, im_weight)
            Data to be reconstructed

        Returns
        -------
        rec : torch.Tensor, shape (batch_size, im_height, im_width)
            Reconstructed image
        """
        with torch.no_grad():
            if Y is None:
                Y_tensor = self.Y_tensor
            else:
                Y_tensor = torch.from_numpy(Y).float().to(self.device)
            x = self.forward(Y_tensor)
            rec = self.convt(x, self.D)

        return rec.detach().to("cpu").numpy()


class Deconvolution(ConvolutionalInpainting):
    def __init__(
        self,
        n_components,
        atom_height,
        atom_width,
        lambd=0.1,
        max_iter=100,
        init_D=None,
        step=1.0,
        algo="fista",
        alpha=None,
        keep_dico=False,
        tol=1e-6,
        rng=None,
        device=None,
    ):
        super().__init__(
            n_components=n_components,
            atom_height=atom_height,
            atom_width=atom_width,
            lambd=lambd,
            max_iter=max_iter,
            init_D=init_D,
            step=step,
            algo=algo,
            alpha=alpha,
            keep_dico=keep_dico,
            tol=tol,
            rng=rng,
            device=device,
        )

        self.kernel = None

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        with torch.no_grad():
            fourier_D = fft.fftn(self.D, axis=(2, 3))
            lip_D = (
                torch.max(
                    torch.max(
                        torch.real(fourier_D * torch.conj(fourier_D)), dim=3
                    )[0],
                    dim=2,
                )[0]
                .sum()
                .item()
            )
            fourier_kernel = fft.fftn(self.kernel, axis=(2, 3))
            lip_kernel = (
                torch.max(
                    torch.max(
                        torch.real(
                            fourier_kernel * torch.conj(fourier_kernel)
                        ),
                        dim=3,
                    )[0],
                    dim=2,
                )[0]
                .sum()
                .item()
            )
            self.lipschitz = lip_D * lip_kernel

            if self.lipschitz == 0:
                self.lipschitz = 1.0

    def cost(self, y, x):
        """
        Compute the LASSO-like convolutional cost.

        Parameters
        ----------
        y : torch.Tensor, shape (batch_size, im_height, im_width)
            Input signal
        x : torch.Tensor,
            shape (batch_size, n_atoms, im_height - atom_height + 1,
                                        im_width - atom_width + 1)
            Sparse codes

        Returns
        -------
        cost : float
            Cost value
        """
        if self.alpha is None:
            dico = self.D
        else:
            dico = self.D * self.window
        rec = self.convt(x, dico)
        diff = self.convt(rec, self.kernel) - y
        l2 = diff.ravel() @ diff.ravel()
        l1 = torch.sum(torch.abs(x))
        cost = 0.5 * l2 + self.lambd * l1

        return cost

    def forward(self, y):
        """
        (F)ISTA-like forward pass.

        Parameters
        ----------
        y : torch.Tensor, shape (batch_size, im_height, im_weight)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape (batch_size, 1, im_height - atom_height + 1,
                                                  im_width - atom_width + 1)
            Approximation of the sparse code associated to y
        """
        batch_size, im_height, im_width = y.shape
        im_height -= self.kernel.shape[2] - 1
        im_width -= self.kernel.shape[3] - 1
        out = torch.zeros(
            (
                batch_size,
                self.n_components,
                im_height - self.atom_height + 1,
                im_width - self.atom_width + 1,
            ),
            dtype=torch.float,
            device=self.device,
        )

        # For FISTA
        t = 1.0
        iterate = out.clone()

        # Computing step and product
        step = 1.0 / self.lipschitz

        for i in range(self.max_iter):
            if self.alpha is None:
                dico = self.D
            else:
                dico = self.D * self.window
            # Keep last iterate for FISTA
            iterate_old = iterate.clone()

            # Gradient descent
            rec = self.convt(self.convt(out, dico), self.kernel)
            diff = self.conv(rec - y, self.kernel)
            gradient = self.conv(diff, dico)
            out = out - step * gradient

            # Thresholding
            thresh = torch.abs(out) - step * self.lambd
            out = F.relu(thresh)

            iterate = out.clone()
            # Apply momentum for FISTA
            if self.algo == "fista":
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
                out = iterate + ((t - 1.0) / t_new) * (iterate - iterate_old)
                t = t_new

        return out

    def fit(self, Y, kernel):
        """
        Training procedure.

        Parameters
        ----------
        Y : ndarray, shape (batch_size, im_height, im_weight)
            Observations to be processed
        kernel : ndarray, shape (kernel_height, kernel_width)
            Convolutional kernel

        Returns
        -------
        loss : float
            Final value of the loss after training
        """
        # Kernel
        self.kernel = torch.from_numpy(kernel).float().to(self.device)
        self.kernel = self.kernel[None, None, :, :]

        # Dictionary
        if self.init_D is None:
            if self.rng is None:
                self.rng = np.random.get_default_rng()
            dico = self.rng.normal(
                size=(self.n_components, 1, self.atom_height, self.atom_width)
            )
            self.D = nn.Parameter(
                torch.tensor(dico, device=self.device, dtype=torch.float)
            )
        else:
            dico_tensor = torch.from_numpy(self.init_D).float().to(self.device)
            self.D = nn.Parameter(dico_tensor)

        # Scaling and computing lipschitz
        self.rescale()
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(Y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss
