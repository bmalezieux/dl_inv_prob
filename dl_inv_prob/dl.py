import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


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
    def __init__(self, n_components, lambd=0.1, max_iter=100,
                 init_D=None, step=1., algo="fista",
                 keep_dico=False, tol=1e-6, rng=None, device=None):
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
        """ Returns the current dictionary 
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
            norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1.
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
        """ Computes an upper bound of the
        Lipschitz constant of the dictionary for each matrix"""
        with torch.no_grad():
            product = torch.matmul(self.operator, self.D)
            self.lipschitz = torch.norm(
                torch.matmul(product.transpose(1, 2), product),
                dim=(1, 2)
            ).max().item()
            if self.lipschitz == 0:
                self.lipschitz = 1.

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
            device=self.device
        )

        # For FISTA
        t = 1.
        iterate = out.clone()

        # Computing step and product
        step = 1. / self.lipschitz
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
                out = iterate + ((t - 1.) / t_new) * (iterate - iterate_old)
                t = t_new

        return out

    def cost(self, y, x):
        """ Cost function """
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
                    self.D -= (beta-1) * t * self.D.grad
                else:
                    init = False

                # Rescaling
                norm_atoms = self.rescale()

                # Computing step
                self.compute_lipschitz()

                # Computing loss with new parameters
                current_cost = self.cost(self.Y_tensor,
                                         self.forward(self.Y_tensor)).item()

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
        future_step = min(10*t, 1e4)
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
                    self.D.grad.data = torch.matmul(self.cov_inv,
                                                    self.D.grad.data)

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
            dico = self.rng.normal(
                size=(self.dim_signal, self.n_components)
            )
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
