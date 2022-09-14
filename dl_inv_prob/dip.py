from dl_inv_prob.common_utils import (
    np_to_torch,
    torch_to_np,
)
from dl_inv_prob.models.skip import SkipNet
from dl_inv_prob.utils import psnr
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F


class DIPInpainting(nn.Module):
    def __init__(
        self,
        model,
        loss=nn.MSELoss(),
        n_iter=1000,
        lr=0.01,
        sigma_input_noise=1.0,
        sigma_reg_noise=0.03,
        input_depth=32,
        output_depth=1,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        if model == "SkipNet":
            self.model = (
                SkipNet(
                    input_channels=input_depth,
                    output_channels=output_depth,
                )
                .to(device)
                .type(dtype)
            )
        self.loss = loss.type(dtype)
        self.n_iter = n_iter
        self.lr = lr
        self.sigma_input_noise = sigma_input_noise
        self.sigma_reg_noise = sigma_reg_noise
        self.input_depth = input_depth
        self.device = device
        self.dtype = dtype

    def fit(self, y, mask, clean=None):
        y = np_to_torch(y).to(self.device).type(self.dtype)
        mask = np_to_torch(mask).to(self.device).type(self.dtype)

        height, width = y.shape[2:]
        self.input_noise = self.sigma_input_noise * torch.rand(
            1, self.input_depth, height, width
        ).to(self.device).type(self.dtype)

        parameters = [p for p in self.model.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for n_iter in range(self.n_iter + 1):
            optimizer.zero_grad()

            input = self.input_noise + self.sigma_reg_noise * torch.randn_like(
                self.input_noise
            ).to(self.device).type(self.dtype)
            out = self.model(input)

            if clean is not None and n_iter % 50 == 0:
                out_np = np.clip(torch_to_np(out), 0, 1)
                psnr_rec = psnr(out_np, clean)

                print(f"Iteration {n_iter}/{self.n_iter}, psnr = {psnr_rec}")

            loss = self.loss(out * mask, y)
            loss.backward()

            optimizer.step()

    def rec(self):
        out = self.model(self.input_noise)

        return np.clip(torch_to_np(out), 0, 1)


class DIPDeblurring(nn.Module):
    def __init__(
        self,
        model,
        loss=nn.MSELoss(),
        n_iter=1000,
        lr=0.01,
        sigma_input_noise=1.0,
        sigma_reg_noise=0.03,
        input_depth=32,
        output_depth=1,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        if model == "SkipNet":
            self.model = (
                SkipNet(
                    input_channels=input_depth,
                    output_channels=output_depth,
                )
                .to(device)
                .type(dtype)
            )
        self.loss = loss.type(dtype)
        self.n_iter = n_iter
        self.lr = lr
        self.sigma_input_noise = sigma_input_noise
        self.sigma_reg_noise = sigma_reg_noise
        self.input_depth = input_depth
        self.device = device
        self.dtype = dtype

    def fit(self, y, blurr, clean=None):
        y = np_to_torch(y).to(self.device).type(self.dtype)
        blurr = np_to_torch(blurr).to(self.device).type(self.dtype)

        height, width = y.shape[2:]
        height -= blurr.shape[2] - 1
        width -= blurr.shape[3] - 1
        self.input_noise = self.sigma_input_noise * torch.rand(
            1, self.input_depth, height, width
        ).to(self.device).type(self.dtype)

        parameters = [p for p in self.model.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for n_iter in range(self.n_iter + 1):
            optimizer.zero_grad()

            input = self.input_noise + self.sigma_reg_noise * torch.randn_like(
                self.input_noise
            ).to(self.device).type(self.dtype)
            out = self.model(input)

            if clean is not None and n_iter % 50 == 0:
                out_np = np.clip(torch_to_np(out), 0, 1)
                psnr_rec = psnr(out_np, clean)

                print(f"Iteration {n_iter}/{self.n_iter}, psnr = {psnr_rec}")

            loss = self.loss(
                F.conv_transpose2d(out, blurr),
                y
            )
            loss.backward()

            optimizer.step()

    def rec(self):
        out = self.model(self.input_noise)

        return np.clip(torch_to_np(out), 0, 1)
