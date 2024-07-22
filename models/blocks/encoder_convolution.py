"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Convolutional encoder architecture for the models.

@references:
unoranic: https://github.com/sdoerrich97/unORANIC

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch.nn as nn

# Import own packages
from .convolution import ResidualBlock, DownBlock


class EncoderConv(nn.Module):
    """
    Encoder with convolutional backbone.
    """

    def __init__(self, img_size: int, in_channel: int, latent_dim: int, dropout: float, feature_maps=16):
        """
        :param img_size: Width and height of the input image.
        :param in_channel: Channel dimension of the input.
        :param latent_dim: Latent dimension encoder and decoder.
        :param dropout: Dropout regularization rate.
        :param feature_maps: Size of feature maps.
        """

        # Initialize the parent constructor
        super().__init__()

        # Store the parameters
        self.img_size = img_size
        self.in_channel = in_channel
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.feature_maps = feature_maps

        if self.img_size == 28:
            self.model = nn.Sequential(
                # input 28 x 28 x C_in
                ResidualBlock(self.in_channel),
                DownBlock(self.in_channel, self.feature_maps, kernel_size=4),
                # input 14 x 14 x 16
                ResidualBlock(self.feature_maps),
                DownBlock(self.feature_maps, self.feature_maps * 2, kernel_size=4),
                nn.Dropout(self.dropout),
                # input 7 x 7 x 32
                ResidualBlock(self.feature_maps * 2),
                DownBlock(self.feature_maps * 2, self.feature_maps * 4, kernel_size=5),
                nn.Dropout(self.dropout),
                # input 3 x 3 x 64
                ResidualBlock(self.feature_maps * 4),
                DownBlock(self.feature_maps * 4, self.latent_dim, kernel_size=5)
                # input 1 x 1 x 256
            )

        elif self.img_size == 224:
            self.model = nn.Sequential(
                # input 224 x 224 x C_in
                ResidualBlock(self.in_channel),
                DownBlock(self.in_channel, self.feature_maps, kernel_size=4),
                # input 112 x 112 x 16
                ResidualBlock(self.feature_maps),
                DownBlock(self.feature_maps, self.feature_maps * 2, kernel_size=4),
                nn.Dropout(self.dropout),
                # input 56 x 56 x 32
                ResidualBlock(self.feature_maps * 2),
                DownBlock(self.feature_maps * 2, self.feature_maps * 4, kernel_size=4),
                nn.Dropout(self.dropout),
                # input 28 x 28 x 64
                ResidualBlock(self.feature_maps * 4),
                DownBlock(self.feature_maps * 4, self.feature_maps * 8, kernel_size=4),
                nn.Dropout(self.dropout),
                # input 14 x 14 x 128
                ResidualBlock(self.feature_maps * 8),
                DownBlock(self.feature_maps * 8, self.feature_maps * 16, kernel_size=4),
                nn.Dropout(self.dropout),
                # input 7 x 7 x 256
                ResidualBlock(self.feature_maps * 16),
                DownBlock(self.feature_maps * 16, self.feature_maps * 32, kernel_size=5),
                nn.Dropout(self.dropout),
                # input 3 x 3 x 512
                ResidualBlock(self.feature_maps * 32),
                DownBlock(self.feature_maps * 32, self.latent_dim, kernel_size=5)
                # input 1 x 1 x 256
            )

        else:
            raise SystemExit

    def forward(self, x):
        """
        Forward Pass of the encoder.

        :param x: Batch of input images.

        :return: Encoded input.
        """

        x = self.model(x)

        return x
