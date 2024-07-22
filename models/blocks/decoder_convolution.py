"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Convolutional decoder architecture for the models.

@references:
unoranic: https://github.com/sdoerrich97/unORANIC

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch.nn as nn

# Import own packages
from .convolution import ResidualBlock, UpBlock


class DecoderConv(nn.Module):
    """
    Decoder with convolutional backbone.
    """

    def __init__(self, img_size: int, out_channel: int, latent_dim: int, dropout: float, feature_maps=16):
        """
        :param img_size: Width and height of the output image.
        :param out_channel: Channel dimension of the output.
        :param latent_dim: Latent dimension encoder and decoder.
        :param dropout: Dropout regularization rate.
        :param feature_maps: Size of feature maps.
        """

        # Initialize the parent constructor
        super().__init__()

        # Store the parameters
        self.img_size = img_size
        self.out_channel = out_channel
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.feature_maps = feature_maps

        if self.img_size == 28:
            self.model = nn.Sequential(
                # input 1 x 1 x 128
                UpBlock(self.latent_dim, self.feature_maps * 4, kernel_size=5),
                ResidualBlock(self.feature_maps * 4),
                nn.Dropout(self.dropout),
                # input 3 x 3 x 64
                UpBlock(self.feature_maps * 4, self.feature_maps * 2, kernel_size=5),
                ResidualBlock(self.feature_maps * 2),
                nn.Dropout(self.dropout),
                # input 7 x 7 x 32
                UpBlock(self.feature_maps * 2, self.feature_maps, kernel_size=4),
                ResidualBlock(self.feature_maps),
                nn.Dropout(self.dropout),
                # input 14 x 14 x 16
                UpBlock(self.feature_maps, self.out_channel, kernel_size=4),
                ResidualBlock(self.out_channel),
                # output 28 x 28 x C_out
            )

        elif self.img_size == 224:
            self.model = nn.Sequential(
                # input 1 x 1 x 256
                UpBlock(self.latent_dim, self.feature_maps * 32, kernel_size=5),
                ResidualBlock(self.feature_maps * 32),
                nn.Dropout(self.dropout),
                # input 3 x 3 x 512
                UpBlock(self.feature_maps * 32, self.feature_maps * 16, kernel_size=5),
                ResidualBlock(self.feature_maps * 16),
                nn.Dropout(self.dropout),
                # input 7 x 7 x 256
                UpBlock(self.feature_maps * 16, self.feature_maps * 8, kernel_size=4),
                ResidualBlock(self.feature_maps * 8),
                nn.Dropout(self.dropout),
                # input 14 x 14 x 128
                UpBlock(self.feature_maps * 8, self.feature_maps * 4, kernel_size=4),
                ResidualBlock(self.feature_maps * 4),
                nn.Dropout(self.dropout),
                # input 28 x 28 x 64
                UpBlock(self.feature_maps * 4, self.feature_maps * 2, kernel_size=4),
                ResidualBlock(self.feature_maps * 2),
                nn.Dropout(self.dropout),
                # input 56 x 56 x 32
                UpBlock(self.feature_maps * 2, self.feature_maps, kernel_size=4),
                ResidualBlock(self.feature_maps),
                nn.Dropout(self.dropout),
                # input 112 x 112 x 16
                UpBlock(self.feature_maps, self.out_channel, kernel_size=4),
                ResidualBlock(self.out_channel),
                # output 224 x 224 x C_out
            )

        else:
            raise SystemExit

    def forward(self, x):
        """
        Forward Pass of the decoder.

        :param x: Latent representation of the input.

        :return: Reconstructed input.
        """

        x = self.model(x)

        return x
