"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Corruption detector for the MedMNIST dataset corrupted with the Albumentations library.

@references:
unoranic: https://github.com/sdoerrich97/unORANIC
albumentations: https://github.com/albumentations-team/albumentations

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch
import torch.nn as nn

# Import own packages
from .vision_transformer import ViT
from .resnet import ResNet18, ResNet50
from .blocks.encoder_convolution import EncoderConv
from .blocks.encoder_vision import EncoderViT
from .blocks.classification import Prediction, IdentityDecoder, SingleLayerDecoder, ThreeLayerDecoder, ViTDecoder


class CorruptionDetector(nn.Module):
    """
    Corruption Detector for the MedMNIST dataset corrupted with the Albumentations library.
    """

    def __init__(self, img_size: int, in_channel: int, latent_dim: int, architecture_encoder: str,
                 architecture_decoder: str, task: str, nr_classes: int, **kwargs):
        """
        :param img_size: Width and height of the input image.
        :param in_channel: Channel dimension of the input.
        :param latent_dim: Latent dimension of the encoder.
        :param architecture_encoder: Which network architecture to use for the encoder.
        :param architecture_decoder: Which network architecture to use for the decoder.
        :param task: What classification task.
        :param nr_classes: Number of classes for the content classification task.
        :param kwargs: Specifications for the used architecture.
            - Vision transformer
                - patch_size: Width and height of the patches.
                - embed_dim: Embedding (latent) dimension of each patch for the encoder and decoder.
                - depth: Depth of the encoder and decoder.
                - num_heads: Number of attention heads for both encoder and decoder.
                - mlp_ratio: Ratio of the hidden dimension compared to the input dimension of the Multi-Layer-Perceptron (MLP) - layer.
                - norm_layer: Normalization layer.
            - Convolutional network
                - dropout: Dropout regularization rate.
        """

        # Run parent constructor
        super().__init__()

        # Store parameters
        self.img_size = img_size
        self.in_channel = in_channel
        self.latent_dim = latent_dim
        self.architecture_encoder = architecture_encoder
        self.architecture_decoder = architecture_decoder
        self.task = task
        self.nr_classes = nr_classes
        self.kwargs = kwargs

        # Initialize the encoder(s)
        if self.architecture_encoder == 'resnet18':
            self.encoder = ResNet18(self.in_channel, self.nr_classes)

        elif self.architecture_encoder == 'resnet50':
            self.encoder = ResNet50(self.in_channel, self.nr_classes)

        elif self.architecture_encoder == 'ViT':
            self.encoder = ViT(self.img_size, self.in_channel, self.kwargs["patch_size"], self.latent_dim,
                               self.kwargs["depth"], self.kwargs["num_heads"], self.kwargs["mlp_ratio"],
                               self.kwargs["norm_layer"])

        elif self.architecture_encoder == 'unoranic+':
            self.encoder = EncoderViT(self.img_size, self.in_channel, self.kwargs["patch_size"], self.latent_dim,
                                      self.kwargs["depth"], self.kwargs["num_heads"], self.kwargs["mlp_ratio"],
                                      self.kwargs["norm_layer"])

        elif self.architecture_encoder == 'unoranic':
            # self.encoder = EncoderConv(self.img_size, self.in_channel, self.latent_dim, self.kwargs["dropout"])
            self.encoder_anatomy = EncoderConv(self.img_size, self.in_channel, self.latent_dim, self.kwargs["dropout"])
            self.encoder_characteristics = EncoderConv(self.img_size, self.in_channel, self.latent_dim, self.kwargs["dropout"])

            # Increase the latent dimension size because the features of both branches are passed to the decoder
            self.latent_dim *= 2

        # Initialize the decoder(s)
        if self.architecture_decoder == 'identity':
            self.decoder = IdentityDecoder()

        elif self.architecture_decoder == 'single_layer':
            if self.architecture_encoder == 'ViT' or self.architecture_encoder == 'unoranic':
                self.decoder = SingleLayerDecoder(self.latent_dim, self.nr_classes)

            elif self.architecture_encoder == 'unoranic+':
                nr_patches = self.encoder.get_num_patches()
                self.decoder = SingleLayerDecoder(self.latent_dim * nr_patches, self.nr_classes)

        elif self.architecture_decoder == 'three_layer':
            if self.architecture_encoder == 'ViT' or self.architecture_encoder == 'unoranic':
                self.decoder = ThreeLayerDecoder(self.latent_dim, self.nr_classes)

            elif self.architecture_encoder == 'unoranic+':
                nr_patches = self.encoder.get_num_patches()
                self.decoder = ThreeLayerDecoder(self.latent_dim * nr_patches, self.nr_classes)

        elif self.architecture_decoder == 'ViT':
            if self.architecture_encoder == 'unoranic+':
                nr_patches = self.encoder.get_num_patches()

            else:
                raise SystemExit("Wrong encoder-decoder setting!")

            self.decoder = ViTDecoder(nr_patches, self.in_channel, self.kwargs["patch_size"], self.latent_dim,
                                      self.kwargs["depth"], self.kwargs["num_heads"], self.kwargs["mlp_ratio"],
                                      self.kwargs["norm_layer"], self.nr_classes)

        # Initialize the predictor
        self.predictor = Prediction(self.task)

    def forward(self, x):
        """
        Forward Pass of the Classifier.

        :param x: Batch of input images.

        :return: Pedictions
        """

        # Run the input through the encoder
        if self.architecture_encoder == 'unoranic':
            Z_anatomy = self.encoder_anatomy(x)
            Z_characteristics = self.encoder_characteristics(x)
            Z = torch.cat((Z_anatomy, Z_characteristics), dim=1)

        else:
            Z = self.encoder(x)

        # Reshape the encoded features to a 2d feature vector for the linear decoder architectures
        if self.architecture_decoder != 'ViT':
            Z = Z.reshape(Z.shape[0], -1)

        # Run the reshaped encoded features through the decoder
        Y = self.decoder(Z)

        # Create the prediction
        Y = self.predictor(Y)

        # Return the classification prediction
        return Y
