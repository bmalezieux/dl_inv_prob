import torch
import torch.nn as nn


class MainBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
        encode=True,
    ):
        """Main convolutional block of the encoder/decoder.

        Parameters
        ----------
        input_channels : int
            Number of input channels
        output_channels : int
            Number of output channels
        kernel_size : int, optional (default: 3)
            Size of the convolution kernel
        act : nn.Module, optional (default: LeakyReLU(negative_slope=0.2,
                                                      inplace=True))
            Activation function
        encode : bool, optional (default: True)
            If True, create an encoding block
        """
        super().__init__()

        padding = (kernel_size - 1) // 2
        if encode:
            stride = 2
        else:
            stride = 1
            self.input_bn = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)

        if not encode:
            kernel_size = 1
            padding = 0
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.act = act
        self.encode = encode

    def forward(self, x):
        """Forward method.

        Parameters
        ----------
        x : torch.Tensor, shape (1, input_channels, height, width)
            Input tensor

        Returns
        -------
        out : torch.Tensor, shape (1, output_channels, out_height, out_width)
            Output tensor
        """
        if self.encode:
            out = x
        else:
            out = self.input_bn(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        return out


class SkipBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=1,
        act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
    ):
        """Convolutional block for skip connections.

        Parameters
        ----------
        input_channels : int
            Number of input channels
        output_channels : int
            Number of output channels
        kernel_size : int, optional (default: 1)
            Size of the convolution kernel
        act : nn.Module, optional (default: LeakyReLU(negative_slope=0.2,
                                                      inplace=True))
            Activation function
        """
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = act

    def forward(self, x):
        """Forward method.

        Parameters
        ----------
        x : torch.Tensor, shape (1, input_channels, height, width)
            Input tensor

        Returns
        -------
        out : torch.Tensor, shape (1, output_channels, out_height, out_width)
            Output tensor
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
    ):
        """Encoding convolutional neural network.

        Parameters
        ----------
        channels : list of int
            Number of output channels for each convolutional block
        kernel_size : int, optional (default: 3)
            Size of the convolution kernels
        act : nn.Module, optional (default: LeakyReLU(negative_slope=0.2,
                                                      inplace=True))
            Activation function
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MainBlock(
                    input_channels=channels[i],
                    output_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    act=act,
                    encode=True,
                )
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x):
        """Forward method.

        Parameters
        ----------
        x : torch.Tensor, shape (1, input_channels, height, width)
            Input tensor

        Returns
        -------
        out : torch.Tensor, shape (1, output_channels, out_height, out_width)
            Output tensor
        """
        out = x
        enc_res = [x]
        for block in self.blocks:
            out = block(out)
            enc_res.append(out)

        return enc_res


class Decoder(nn.Module):
    def __init__(
        self,
        channels,
        channels_enc,
        channels_skip,
        kernel_size=3,
        kernel_size_skip=1,
        act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
    ):
        """Decoding convolutional neural network with skip connections.

        Parameters
        ----------
        channels : list of int
            Number of output channels of each convolutional block
        kernel_size : int, optional (default: 3)
            Size of the convolution kernels
        act : nn.Module, optional (default: LeakyReLU(negative_slope=0.2,
                                                      inplace=True))
            Activation function
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MainBlock(
                    input_channels=channels[i] + channels_skip[i],
                    output_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    act=act,
                    encode=False,
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.skip_blocks = nn.ModuleList(
            [
                SkipBlock(
                    input_channels=channels_enc[::-1][i + 1],
                    output_channels=channels_skip[i],
                    kernel_size=kernel_size_skip,
                    act=act,
                )
                for i in range(len(channels_skip))
            ]
        )

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, enc_out, enc_res):
        """Forward method.

        Parameters
        ----------
        enc_out : torch.Tensor, shape (1, input_channels, height, width)
            Output tensor of the encoder
        enc_res : list of torch.Tensor
            Intermediate results of the encoding network.

        Returns
        -------
        out : torch.Tensor, shape (1, output_channels, out_height, out_width)
            Output tensor
        """
        out = enc_out
        for block, skip_block, enc in zip(
            self.blocks, self.skip_blocks, enc_res
        ):
            out = self.upsample(out)
            skip_res = skip_block(enc)
            out = torch.concat((skip_res, out), dim=1)
            out = block(out)

        return out


class SkipNet(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        channels_enc=[128] * 3,
        channels_dec=[128] * 3,
        channels_skip=[128] * 3,
        kernel_size_enc=3,
        kernel_size_dec=3,
        kernel_size_skip=1,
        act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
    ):
        """U-Net like convolutional neural network.

        https://box.skoltech.ru/index.php/s/INaUzvTWLak3h7Q#pdfviewer

        Parameters
        ----------
        input_channels : int
            Number of input channels
        output_channels : int
            Number of output channels
        channels_enc : list of int
            Number of output channels of encoding blocks
        channels_dec : list of int
            Number of output channels of decoding blocks
        channels_skip : list of int
            Number of output channels of skip blocks
        kernel_size_enc : int, optional (default: 3)
            Size of the convolution kernels of encoding blocks
        kernel_size_dec : int, optional (default: 3)
            Size of the convolution kernels of decoding blocks
        kernel_size _skip: int, optional (default: 1)
            Size of the convolution kernels of skip blocks
        act : nn.Module, optional (default: LeakyReLU(negative_slope=0.2,
                                                      inplace=True))
            Activation function
        """
        super().__init__()

        self.encoder = Encoder(
            channels=[input_channels] + channels_enc,
            kernel_size=kernel_size_enc,
            act=act,
        )
        self.decoder = Decoder(
            channels=[channels_enc[-1]] + channels_dec,
            channels_enc=[input_channels] + channels_enc,
            channels_skip=channels_skip,
            kernel_size=kernel_size_dec,
            kernel_size_skip=kernel_size_skip,
            act=act,
        )
        self.last_layer = nn.Conv2d(channels_dec[-1], output_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward method.

        Parameters
        ----------
        x : torch.Tensor, shape (1, input_channels, height, width)
            Input tensor

        Returns
        -------
        out : torch.Tensor, shape (1, output_channels, height, width)
            Output tensor
        """
        *enc_res, out = self.encoder(x)
        out = self.decoder(out, enc_res[::-1])
        out = self.last_layer(out)
        out = self.sigmoid(out)

        return out
