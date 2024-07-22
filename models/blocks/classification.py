"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Decoder architectures for the classifier models.

@references:
unoranic: https://github.com/sdoerrich97/unORANIC

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

# Import own packages
from ...utils import PositionEmbedding


class Prediction(nn.Module):
    """
    Prediction block for a classification task.
    """

    def __init__(self, task: str):
        """
        :param task: What classification task.
        """

        # Run parent constructor
        super().__init__()

        # Initialize the model
        if task == 'multi-label, binary-class':
            self.model = nn.Sigmoid()

        else:
            self.model = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Prediction.

        :param x: Input.

        :return: Prediction.
        """

        return self.model(x)


class IdentityDecoder(nn.Module):
    """
    Identity decoder for a classification task.
    """

    def __init__(self):
        # Run parent constructor
        super().__init__()

        # Initialize the model
        self.model = nn.Identity()

    def forward(self, x):
        """
        Prepate the encoded input x for prediction.

        :param x: Encoded input.
        """

        return self.model(x)


class SingleLayerDecoder(nn.Module):
    """
    Single layer decoder for a classification task.
    """

    def __init__(self, latent_dim: int, nr_classes: int):
        """
        :param latent_dim: Latent dimension of the encoder.
        :param nr_classes: Number of classes for the content classification task.
        """

        # Run parent constructor
        super().__init__()

        # Initialize the model
        self.model = nn.Linear(latent_dim, nr_classes)

    def forward(self, x):
        """
        Prepate the encoded input x for prediction.

        :param x: Encoded input.
        """

        return self.model(x)


class ThreeLayerDecoder(nn.Module):
    """
    Three layer decoder for a classification task.
    """

    def __init__(self, latent_dim: int, nr_classes: int):
        """
        :param latent_dim: Latent dimension of the encoder.
        :param nr_classes: Number of classes for the content classification task.
        """

        # Run parent constructor
        super().__init__()

        # Initialize the model
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.Linear(latent_dim // 4, nr_classes)
        )

    def forward(self, x):
        """
        Prepate the encoded input x for prediction.

        :param x: Encoded input.
        """

        return self.model(x)


class ViTDecoder(nn.Module):
    """
    Decoder with VisionTransformer backbone for a classification task.
    """

    def __init__(self, num_patches: int, in_channel: int, patch_size: int, embed_dim: int, depth: int, num_heads: int,
                 mlp_ratio: float, norm_layer: str, nr_classes: int):
        """

        :param num_patches: Number of patches.
        :param in_channel: Channel dimension of the input.
        :param patch_size: Width and height of the patches.
        :param embed_dim: Embedding (latent) dimension for the decoder.
        :param depth: Depth.
        :param num_heads: Number of attention heads.
        :param mlp_ratio: Ratio of the hidden dimension compared to the input dimension of the Multi-Layer-Perceptron (MLP) - layer.
        :param norm_layer: Normalization layer.
        :param nr_classes: Number of classes for the content classification task.
        """

        # Initialize the parent constructor
        super().__init__()

        # Store the parameters
        self.num_patches = num_patches
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.nr_classes = nr_classes

        # Initialize the normalization layer
        if self.norm_layer == 'nn.LayerNorm':
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Create the embedding
        self.patch_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Create the position embedding (self.num_patches + 1 has to be used when cls_token=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False)

        # Create the attention blocks
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for i in range(self.depth)])

        # Create the normalization layer
        self.norm = self.norm_layer(self.embed_dim)

        # Create the prediction layer to map the latent_dim to the number of classes for the classification task
        self.pred = nn.Linear(self.embed_dim, self.nr_classes)

        # Initialize the decoder's weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the decoder.
        """

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = PositionEmbedding.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize nn.Linear and nn.LayerNorm.

        :param m: Module to initialize
        """

        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, X):
        """
        Forward pass of the decoder.

        :param X: Latent representation of the input.

        :return: Reconstructed input.
        """

        # Embed the tokens
        X = self.patch_embed(X)

        # Append cls token
        cls_tokens = self.cls_token.expand(X.shape[0], -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)

        # Add pos embed
        X = X + self.pos_embed

        # Apply Transformer blocks
        for blk in self.blocks:
            X = blk(X)

        # Apply the normalization
        X = self.norm(X)

        # Map the latent_dimension to the number of classes
        X = self.pred(X)

        # Extract the prediction
        Y = X[:, 0]

        return Y
