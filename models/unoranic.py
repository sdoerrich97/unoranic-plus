"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
unoranic with convolutional backbone.

@references:
unoranic: https://github.com/sdoerrich97/unORANIC

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
from functools import partial
import torch
import torch.nn as nn

# Import own packages
from .blocks.encoder_convolution import EncoderConv
from .blocks.decoder_convolution import DecoderConv
from ..utils import Loss


class Unoranic(nn.Module):
    """
    unoranic with convolutional backbone.
    """

    def __init__(self, img_size: int, in_channel: int, latent_dim: int, dropout: float):
        """
        :param img_size: Width and height of the input image.
        :param in_channel: Channel dimension of the input.
        :param latent_dim: Latent dimension encoder and decoder.
        :param dropout: Dropout regularization rate.
        """

        # Initialize the parent constructor
        super().__init__()

        # Store the parameters
        self.img_size = img_size
        self.in_channel = in_channel
        self.latent_dim = latent_dim
        self.dropout = dropout

        # Create the encoder
        self.encoder_anatomy = EncoderConv(self.img_size, self.in_channel, self.latent_dim, self.dropout)

        self.encoder_characteristics = EncoderConv(self.img_size, self.in_channel, self.latent_dim, self.dropout)

        # Create the decoder
        self.decoder_anatomy = DecoderConv(self.img_size, self.in_channel, self.latent_dim, self.dropout)

        self.decoder = DecoderConv(self.img_size, self.in_channel, 2 * self.latent_dim, self.dropout)

    def forward(self, X_orig, X, X_anatomy, X_characteristics):
        """
        Forward pass of the model.

        :param X_orig: Batch of original images.
        :param X: Batch of input images.
        :param X_anatomy: Image tuple for the anatomy branch.
        :param X_characteristics: Image tuple for the characteristics branch.

        :return: Loss, metrics and reconstructions
        """

        batch_size = X.shape[0]

        # Pass the input through the anatomy encoder to get the anatomical latent representation
        Z_anatomy = self.encoder_anatomy(X)
        Z_anatomy = Z_anatomy.reshape(batch_size, -1)

        # Pass the input through the characteristics encoder to get the characteristics latent representation
        Z_characteristics = self.encoder_characteristics(X)
        Z_characteristics = Z_characteristics.reshape(batch_size, -1)

        # Concatenate the feature embeddings patch-wise
        Z = torch.cat((Z_anatomy, Z_characteristics), dim=1)

        # Pass the anatomy latent representation through the anatomy decoder to get an anatomy reconstruction
        X_hat_anatomy = self.decoder_anatomy(Z_anatomy.reshape(batch_size, self.latent_dim, 1, 1))

        # Pass the combined latent representation through the image decoder to reconstruct the input image
        X_hat = self.decoder(Z.reshape(batch_size, 2 * self.latent_dim, 1, 1))

        # Calculate the difference images
        X_diff = torch.sub(X, X_orig)  # Difference of original image to its (corrupted) input image version
        X_hat_diff = torch.sub(X_hat, X_hat_anatomy)  # Difference of reconstructed image to the anatomy reconstruction

        # Calculate the loss and the other metrics
        mse_input, ssim_input, psnr_input = Loss.calculate_reconstruction_loss(X, X_hat)  # Input image
        mse_anatomy, ssim_anatomy, psnr_anatomy = Loss.calculate_reconstruction_loss(X_orig, X_hat_anatomy)  # Anatomy
        mse_diff, ssim_diff, psnr_diff = Loss.calculate_reconstruction_loss(X_diff, X_hat_diff)  # Difference images

        # Run the input through the embeddings consistency loss
        if X_anatomy:
            consistency_loss_anatomy, mean_cos_dist_anatomy = \
                Loss.compute_embedding_consistency(X_anatomy, self.encoder_anatomy)

            consistency_loss_characteristics, mean_cos_dist_characteristics = \
                Loss.compute_embedding_consistency(X_characteristics, self.encoder_characteristics)

        else:
            consistency_loss_anatomy, mean_cos_dist_anatomy = None, None
            consistency_loss_characteristics, mean_cos_dist_characteristics = None, None

        return (mse_input, mse_anatomy, mse_diff), (ssim_input, ssim_anatomy, ssim_diff), \
            (psnr_input, psnr_anatomy, psnr_diff), (consistency_loss_anatomy, consistency_loss_characteristics),\
            (mean_cos_dist_anatomy, mean_cos_dist_characteristics), \
            (X_diff.detach().cpu(), X_hat.detach().cpu(), X_hat_anatomy.detach().cpu(), X_hat_diff.detach().cpu())
