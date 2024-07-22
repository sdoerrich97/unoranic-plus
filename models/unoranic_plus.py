"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
unoranic+ with VisionTransformer backbone.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch
import torch.nn as nn

# Import own packages
from .blocks.encoder_vision import EncoderViT
from .blocks.decoder_vision import DecoderViT
from ..utils import Loss, Patches


class UnoranicPlus(nn.Module):
    """
    unoranic+ with VisionTransformer backbone
    """

    def __init__(self, img_size: int, in_channel: int, patch_size: int, embed_dim: int, depth: int, num_heads: int,
                 mlp_ratio: float, norm_layer: str):
        """
        :param img_size: Width and height of the input image.
        :param in_channel: Channel dimension of the input.
        :param patch_size: Width and height of the patches.
        :param embed_dim: Embedding (latent) dimension of each patch for the encoder and decoder.
        :param depth: Depth of the encoder and decoder.
        :param num_heads: Number of attention heads for both encoder and decoder.
        :param mlp_ratio: Ratio of the hidden dimension compared to the input dimension of the Multi-Layer-Perceptron (MLP) - layer.
        :param norm_layer: Normalization layer.
        """

        # Initialize the parent constructor
        super().__init__()

        # Store the parameters
        self.img_size = img_size
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        # Create the encoder
        self.encoder = EncoderViT(self.img_size, self.in_channel, self.patch_size, self.embed_dim, self.depth,
                                          self.num_heads, self.mlp_ratio, self.norm_layer)

        # Create the decoder
        self.num_patches = self.encoder.get_num_patches()
        self.decoder_anatomy = DecoderViT(self.num_patches, self.in_channel, self.patch_size, self.embed_dim,
                                          self.depth, self.num_heads, self.mlp_ratio, self.norm_layer)

        self.decoder = DecoderViT(self.num_patches, self.in_channel, self.patch_size, self.embed_dim, self.depth,
                                  self.num_heads, self.mlp_ratio, self.norm_layer)

    def forward(self, X_orig, X):
        """
        Forward pass of the model.

        :param X_orig: Batch of original images.
        :param X: Batch of input images.

        :return: Loss, metrics and reconstructions
        """
        # Extract the current device
        device = X.device

        # Run the input through the encoder to get the latent representation
        Z = self.encoder(X)

        # Run the latent representation through the decoder to get the reconstructions
        X_hat = self.decoder(Z)  # [B, P, patch_size * patch_size * 3]
        X_hat_anatomy = self.decoder_anatomy(Z)  # [B, P, patch_size * patch_size * 3]

        # Calculate the difference images
        X_diff = torch.sub(X, X_orig)  # Difference of original image to its (corrupted) input image version
        X_hat_diff = torch.sub(X_hat, X_hat_anatomy)  # Difference of reconstructed image to an artifical reconstruction of the uncorrupted input

        # Calculate the loss, the other metrics and unpatchify the reconstructions to their original image dimensions
        mse_input, ssim_input, psnr_input, X_hat = self.compute_loss(X, X_hat)  # Input image
        mse_anatomy, ssim_anatomy, psnr_anatomy, X_hat_anatomy = self.compute_loss(X_orig, X_hat_anatomy)  # Anatomy
        mse_diff, ssim_diff, psnr_diff, X_hat_diff = self.compute_loss(X_diff, X_hat_diff)  # Difference images

        # Return the loss, the other metrics and the reconstructions
        return (mse_input, mse_anatomy, mse_diff), (ssim_input, ssim_anatomy, ssim_diff), \
            (psnr_input, psnr_anatomy, psnr_diff), \
            (torch.tensor(0, dtype=torch.float, device=device), torch.tensor(0, dtype=torch.float, device=device)), \
            (0, 0), \
            (X_diff.detach().cpu(), X_hat.detach().cpu(), X_hat_anatomy.detach().cpu(), X_hat_diff.detach().cpu())

    def compute_loss(self, X, X_hat):
        """
        Compute the loss.

        :param X: Batch of input images of shape [B, C, H, W].
        :param X_hat: Batch of reconstructed images of shape [B, P, patch_size * patch_size * 3].
        """

        # Unpatchify the predictions
        X_hat = Patches.unpatchify(X_hat, self.in_channel, self.patch_size)

        # Calculate the MSE, SSIM and PSNR values for the patchified input X and its reconstruction X_hat
        mse, ssim, psnr = Loss.calculate_reconstruction_loss(X, X_hat)

        return mse, ssim, psnr, X_hat
