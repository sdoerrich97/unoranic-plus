"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Helper blocks for the convolutional models.

@references:
unoranic: https://github.com/sdoerrich97/unORANIC

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch.nn as nn


class DownBlock(nn.Module):
    """Convolution block to decrease image dimension by simultaneously increasing channel dimension"""

    def __init__(self, in_channels, out_channels, kernel_size=4):
        super().__init__()

        # input W x H x C_in
        self.downblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )
        # output W//2 x H//2 x C_out

    def forward(self, x):
        return self.downblock(x)


class UpBlock(nn.Module):
    """Convolution block to decrease image dimension by simultaneously increasing channel dimension"""

    def __init__(self, in_channels, out_channels, kernel_size=4):
        super().__init__()

        # input W x H x C_in
        self.upblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )
        # output 2*W x 2*H x C_out

    def forward(self, x):
        return self.upblock(x)


class ResidualBlock(nn.Module):
    """Residual block containing two times (Convolution => [BN] => ReLU)"""

    def __init__(self, in_channels):
        super().__init__()

        # input W x H x C_in
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )
        # output W x H x C_in

    def forward(self, x):
        return self.resblock(x) + x
