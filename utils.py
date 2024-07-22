"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Helpers for the model creation.

@references:
mae: https://github.com/facebookresearch/mae

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances


class PositionEmbedding:
    """
    Position embedding utils.

    References:
        - MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
        - Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
        - MoCo v3: https://github.com/facebookresearch/moco-v3
        - DeiT: https://github.com/facebookresearch/deit
    """

    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token=False):
        """
        2D sine-cosine position embedding.

        :param embed_dim: Embedding dimension.
        :param grid_size: Grid height and width.

        :return: Position embedding [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = PositionEmbedding.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

        return pos_embed

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        """
        Get the 2d sin-cosine position embedding from the provided grid.

        :param embed_dim: Embedding dimension.
        :param grid:

        :return: 2d sin-cosine position embedding
        """

        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = PositionEmbedding.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = PositionEmbedding.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)

        return emb

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """
        Get the 1d sin-cosine position embedding from the provided grid.

        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)

        :return: 1d sin-cosine position embedding.
        """

        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    @staticmethod
    def interpolate_pos_embed(model, checkpoint_model):
        """
        Interpolate position embeddings for high-resolution.

        :param model:
        :param checkpoint_model:
        :return:
        """

        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed


class Patches:
    """
    Patches utils.
    """

    @staticmethod
    def patchify(X, in_channel, patch_size):
        """
        Split each image of the input batch into patches.

        :param in_channel: Channel dimension of the input.
        :param X: Input batch of images of shape: (N, 3, H, W).

        :return: Patchified batch of images of shape: (N, L, patch_size**2 *3).
        """

        # Assert that the width and height of the input images is the same and the image can be split into
        # non-overlapping patches
        assert X.shape[2] == X.shape[3] and X.shape[2] % patch_size == 0

        # Split the input images into equally sized patches
        h = w = X.shape[2] // patch_size
        X = X.reshape(shape=(X.shape[0], in_channel, h, patch_size, w, patch_size))
        X = torch.einsum('nchpwq->nhwpqc', X)
        X = X.reshape(shape=(X.shape[0], h * w, patch_size ** 2 * in_channel))

        # Return the patchified batch
        return X

    @staticmethod
    def unpatchify(X, in_channel, patch_size):
        """
        Redo a patched input to obtain the original batch of images.

        :param in_channel: Channel dimension of the input.
        :param X: Patchified batch of images of shape: (N, L, patch_size**2 *3).

        :return: Original un-patchified input batch of shape: (N, 3, H, W).
        """

        # Assert the correct height and width of the original input size
        h = w = int(X.shape[1] ** .5)
        assert h * w == X.shape[1]

        # Reshape the patchified input batch to the original image shape
        X = X.reshape(shape=(X.shape[0], h, w, patch_size, patch_size, in_channel))
        X = torch.einsum('nhwpqc->nchpwq', X)
        X = X.reshape(shape=(X.shape[0], in_channel, h * patch_size, h * patch_size))

        # Return the original image-sized batch
        return X


class LearningRate:
    """
    Learning rate utils.
    """

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, lr, min_lr, epochs, warmup_epochs):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < warmup_epochs:
            lr = lr * epoch / warmup_epochs
        else:
            lr = min_lr + (lr - min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class Loss:
    """
    Loss function utils.
    """

    @staticmethod
    def calculate_reconstruction_loss(X, X_hat):
        """
        Calculate the MSE and SSIM loss as well as the PSNR for the given input image(s) and their reconstruction(s).

        :param X: Input image(s).
        :param X_hat: Reconstruction(s).
        :return: MSE, SSIM and PSNR.
        """

        # Calcualte the ssim value
        ssim = structural_similarity_index_measure(X_hat, X, reduction='elementwise_mean')

        # Calculate the mse value
        mse = F.mse_loss(X_hat, X)

        # Calculate the psnr value
        if mse < 1e-15:
            psnr = 20 * torch.log10(torch.tensor(1).to(X.device))

        else:
            psnr = 20 * torch.log10(torch.tensor(1).to(X.device)) - 10 * torch.log10(mse)

        # return the values
        return mse, ssim, psnr

    @staticmethod
    def compute_embedding_consistency(X_tuple, encoder):
        """
        Compute the mean cosine distances and the consistency loss between the embeddings.

        :param X_tuple: Image tuple.
        :param encoder: Which encoder to use for creating the embeddings.

        :return: The consistency loss and the mean cosine distance of the embeddings.
        """

        # Store the embeddings and predictions for each input image
        embeddings = []

        # Pass each input through the network
        for X in X_tuple:
            embed = encoder(X)
            embed = embed.reshape(embed.shape[0], -1)
            embeddings.append(embed)

        # Calculate the consistency loss for each embedding combination
        mean_cos_dist, consistency_loss = 0, 0

        for E1, E2 in combinations(embeddings, 2):
            # Return NaN if there are nans in the embeddings
            if torch.isnan(E1).any() or torch.isnan(E2).any():
                return float('nan'), float('nan')

            # Calculate the cosine distance between the current embedding pair
            mean_cos_dist += np.mean(cosine_distances(E1.detach().cpu(), E2.detach().cpu()))

            # Calculate the mean squared error between the current embedding pair
            consistency_loss += F.mse_loss(E1, E2)

        # Average the consistency loss over the number of embedding combinations
        nr_comb_embed = math.comb(len(embeddings), 2)
        mean_cos_dist = (1 / nr_comb_embed) * mean_cos_dist
        consistency_loss = (1 / nr_comb_embed) * consistency_loss

        # Return the consistency loss
        return consistency_loss, mean_cos_dist


class Metrics:
    """
    Performance metrics.
    """

    @staticmethod
    def getAUC(y_true, y_score, task):
        """
        AUC metric.

        :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        :param y_score: the predicted score of each class,
        shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        :param task: the task of current dataset
        """
        y_true = y_true.squeeze()
        y_score = y_score.squeeze()

        if task == 'multi-label, binary-class':
            auc = 0
            for i in range(y_score.shape[1]):
                label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
                auc += label_auc
            ret = auc / y_score.shape[1]

        elif task == 'binary-class':
            if y_score.ndim == 2:
                y_score = y_score[:, -1]
            else:
                assert y_score.ndim == 1
            ret = roc_auc_score(y_true, y_score)

        else:
            auc = 0
            for i in range(y_score.shape[1]):
                y_true_binary = (y_true == i).astype(float)
                y_score_binary = y_score[:, i]

                if len(np.unique(y_true_binary)) > 1:
                    auc += roc_auc_score(y_true_binary, y_score_binary)

            ret = auc / y_score.shape[1]

        return ret

    @staticmethod
    def getACC(y_true, y_score, task, threshold=0.5):
        """
        Accuracy metric.

        :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        :param y_score: the predicted score of each class,
        shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        :param task: the task of current dataset
        :param threshold: the threshold for multilabel and binary-class tasks
        """
        y_true = y_true.squeeze()
        y_score = y_score.squeeze()

        if task == 'multi-label, binary-class':
            y_pre = y_score > threshold
            acc = 0
            for label in range(y_true.shape[1]):
                label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
                acc += label_acc
            ret = acc / y_true.shape[1]

        elif task == 'binary-class':
            if y_score.ndim == 2:
                y_score = y_score[:, -1]
            else:
                assert y_score.ndim == 1
            ret = accuracy_score(y_true, y_score > threshold)

        else:
            ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

        return ret


class Plotting:
    """
    Plotting utils.
    """

    # Set the plot style
    plt.rcParams['text.usetex'] = True
    plt.style.use('seaborn-paper')

    sns.set_style('white')
    sns.set_context("paper")

    @staticmethod
    def plot_and_save_img(img: np.array, sample_name: str, output_path: Path, grayscale: bool):
        """
        Save the given image as a file.

        :param img: Image.
        :param sample_name: Filename of the image.
        :param output_path: Where to save the image to.
        :param grayscale: Whether to save the image in grayscale or not.
        """

        if img is not None:
            # Plot the image with pyplot
            if grayscale:
                plt.imshow(img, cmap='gray')

            else:
                plt.imshow(img)

            # Remove the axis
            plt.axis("off")

            # Save the image
            plt.savefig(output_path / f"{sample_name}.png", bbox_inches='tight', pad_inches=0, dpi=300)
            plt.savefig(output_path / f"{sample_name}.pdf", bbox_inches='tight', pad_inches=0, dpi=300)

    @staticmethod
    def plot_and_save_images_combined(X_orig: np.array, X: np.array, X_diff: np.array, X_hat: np.array,
                                      X_hat_anatomy: np.array, X_hat_diff:np.array, sample_name: str,
                                      output_path: Path, grayscale: bool):
        """
        Save the given images in a single file.

        :param X_orig: Original image
        :param X: Input (potentially corrupted) image.
        :param X_diff: Difference image of the original to the input image.
        :param X_hat: Reconstructed image.
        :param X_hat_anatomy: Anatomical reconstructed image (only for unoranic!).
        :param X_hat_diff: Difference between the anatomical and the reconstructed image.
        :param sample_name: Filename of the image.
        :param output_path: Where to save the image to.
        :param grayscale: Whether to save the image in grayscale or not.
        """

        # Save all images combined in one figure
        fig, axes = plt.subplots(nrows=2, ncols=3)

        # Assign the correct color map for grayscale and color images
        if grayscale:
            cmap = 'gray'

        else:
            cmap = 'viridis'

        # Plot the images
        axes[0, 0].imshow(X_orig, cmap=cmap)
        axes[0, 0].set_title("Uncorrupted\nImage (I)", loc='center')
        axes[0, 0].axis("off")

        axes[0, 1].imshow(X, cmap=cmap)
        axes[0, 1].set_title("Corrupted\nInput (S)", loc='center')
        axes[0, 1].axis("off")

        axes[0, 2].imshow(X_diff, cmap=cmap)
        axes[0, 2].set_title("Difference\nInputs (I-S)", loc='center')
        axes[0, 2].axis("off")

        axes[1, 1].imshow(X_hat, cmap=cmap)
        axes[1, 1].set_title("Corrupted\nReconstruction ($\hat{S}$)", loc='center')
        axes[1, 1].axis("off")

        if X_hat_anatomy is not None:
            axes[1, 0].imshow(X_hat_anatomy, cmap=cmap)
            axes[1, 0].set_title("Anatomy\nReconstruction ($\hat{I}_A$)", loc='center')
            axes[1, 0].axis("off")

            axes[1, 2].imshow(X_hat_diff, cmap=cmap)
            axes[1, 2].set_title("Difference\nReconstructions ($\hat{I}_A - \hat{S}$)", loc='center')
            axes[1, 2].axis("off")

        # Save and close the figure
        fig.tight_layout(h_pad=2)
        fig.savefig(output_path / f"{sample_name}.png", bbox_inches='tight', pad_inches=0, dpi=300)
        fig.savefig(output_path / f"{sample_name}.pdf", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

    @staticmethod
    def save_img_and_reconstructed_image(X_orig: np.array, X: np.array, X_diff: np.array, X_hat: np.array,
                                         X_hat_anatomy: np.array, X_hat_diff:np.array, X_name, output_path):
        """
        Save the processed input image together with the reconstructed image.

        :param X_orig: Original image
        :param X: Input (potentially corrupted) image.
        :param X_diff: Difference image of the original to the input image.
        :param X_hat: Reconstructed image.
        :param X_hat_anatomy: Anatomical reconstructed image (only for unoranic!).
        :param X_hat_diff: Difference between the anatomical and the reconstructed image.
        :param X_name: Sample name of the original image.
        :param output_path: Where the images shall be stored.
        """

        # Get the sample name
        sample_name = X_name
        channel_dim = X_orig.shape[0]

        # Remove the batch dimension and reorder the and height, width and channel dimensions
        X_orig = X_orig.permute(1, 2, 0).numpy()
        X = X.permute(1, 2, 0).numpy()
        X_diff = X_diff.permute(1, 2, 0).numpy()
        X_hat = X_hat.permute(1, 2, 0).numpy()

        # Normalize the pixel values in the range 0 to 255
        X_orig = cv2.normalize(X_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X = cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X_diff = cv2.normalize(X_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X_hat = cv2.normalize(X_hat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Do the same for the anatomy reconstruction and the reconstruction-difference image if they exist
        if X_hat_anatomy is not None:
            X_hat_anatomy = X_hat_anatomy.permute(1, 2, 0).numpy()
            X_hat_diff = X_hat_diff.permute(1, 2, 0).numpy()

            X_hat_anatomy = cv2.normalize(X_hat_anatomy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            X_hat_diff = cv2.normalize(X_hat_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Save the images in grayscale
        if channel_dim == 1:
            # Save the individual images to PNG files
            Plotting.plot_and_save_img(X_orig, f"{sample_name}_orig", output_path, True)
            Plotting.plot_and_save_img(X, f"{sample_name}_corrupted", output_path, True)
            Plotting.plot_and_save_img(X_diff, f"{sample_name}_difference", output_path, True)
            Plotting.plot_and_save_img(X_hat, f"{sample_name}_reconstructed", output_path, True)
            Plotting.plot_and_save_img(X_hat_anatomy, f"{sample_name}_reconstructed_anatomy", output_path, True)
            Plotting.plot_and_save_img(X_hat_diff, f"{sample_name}_reconstructed_difference", output_path, True)

            # Save all images combined in one figure
            Plotting.plot_and_save_images_combined(X_orig, X, X_diff, X_hat, X_hat_anatomy, X_hat_diff,
                                                   f"{sample_name}_all", output_path, True)

        # Save the images as color images
        else:
            # Save the individual images to PNG files
            Plotting.plot_and_save_img(X_orig, f"{sample_name}_orig", output_path, False)
            Plotting.plot_and_save_img(X, f"{sample_name}_corrupted", output_path, False)
            Plotting.plot_and_save_img(X_diff, f"{sample_name}_difference", output_path, False)
            Plotting.plot_and_save_img(X_hat, f"{sample_name}_reconstructed", output_path, False)
            Plotting.plot_and_save_img(X_hat_anatomy, f"{sample_name}_reconstructed_anatomy", output_path, False)
            Plotting.plot_and_save_img(X_hat_diff, f"{sample_name}_reconstructed_difference", output_path, False)

            # Save all images combined in one figure
            Plotting.plot_and_save_images_combined(X_orig, X, X_diff, X_hat, X_hat_anatomy, X_hat_diff,
                                                   f"{sample_name}_all", output_path, False)

    @staticmethod
    def plot_image_and_corrupted_variants(mode: str, corruptions: list, severity: int, X_orig: torch.Tensor, corrupted_variants: list,
                                          nr_channels: int, output_path: Path):
        """
        Save the given corrupted images in a single file and individually.

        :param mode: Images for train, val or test set.
        :param corruptions: Corruptions.
        :param severity: How strong should the corruption is.
        :param X_orig: Original, uncorrupted images for each set.
        :param corrupted_variants: Corrupted images.
        :param nr_channels: 1 (grayscale) or 3 (RGB) channel images.
        :param output_path: Where to save the image to.
        """

        # Create an output folder for the original image
        (output_path / f"{mode}" / "original").mkdir(parents=True, exist_ok=True)

        # Reorder height, width and channel dimensions for each variant and normalize the pixel values
        X_orig = cv2.normalize(X_orig.permute(1, 2, 0).numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        image_variants = []
        for corruption, variant in zip(corruptions, corrupted_variants):
            # Create an output folder for the current corruption
            (output_path / f"{mode}" / f"{corruption}").mkdir(parents=True, exist_ok=True)

            # Reorder and normalize the variants
            reordered_variant = variant.permute(1, 2, 0).numpy()
            normalized_variant = cv2.normalize(reordered_variant, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            image_variants.append(normalized_variant)

        # Assign the correct color map for grayscale and color images
        if nr_channels == 1:
            grayscale = True
            cmap = 'gray'

        else:
            grayscale = False
            cmap = 'viridis'

        # Save each corrupted variant individually
        Plotting.plot_and_save_img(X_orig, f"{mode}_original", output_path / mode / "original", grayscale)
        Plotting.plot_and_save_img(image_variants[0], f"{mode}_{corruptions[0]}_{severity}", output_path / mode / corruptions[0], grayscale)
        Plotting.plot_and_save_img(image_variants[1], f"{mode}_{corruptions[1]}_{severity}", output_path / mode / corruptions[1], grayscale)
        Plotting.plot_and_save_img(image_variants[2], f"{mode}_{corruptions[2]}_{severity}", output_path / mode / corruptions[2], grayscale)
        Plotting.plot_and_save_img(image_variants[3], f"{mode}_{corruptions[3]}_{severity}", output_path / mode / corruptions[3], grayscale)
        Plotting.plot_and_save_img(image_variants[4], f"{mode}_{corruptions[4]}_{severity}", output_path / mode / corruptions[4], grayscale)
        Plotting.plot_and_save_img(image_variants[5], f"{mode}_{corruptions[5]}_{severity}", output_path / mode / corruptions[5], grayscale)
        Plotting.plot_and_save_img(image_variants[6], f"{mode}_{corruptions[6]}_{severity}", output_path / mode / corruptions[6], grayscale)
        Plotting.plot_and_save_img(image_variants[7], f"{mode}_{corruptions[7]}_{severity}", output_path / mode / corruptions[7], grayscale)
        Plotting.plot_and_save_img(image_variants[8], f"{mode}_{corruptions[8]}_{severity}", output_path / mode / corruptions[8], grayscale)
        Plotting.plot_and_save_img(image_variants[9], f"{mode}_{corruptions[9]}_{severity}", output_path / mode / corruptions[9], grayscale)
        Plotting.plot_and_save_img(image_variants[10], f"{mode}_{corruptions[10]}_{severity}", output_path / mode / corruptions[10], grayscale)
        Plotting.plot_and_save_img(image_variants[11], f"{mode}_{corruptions[11]}_{severity}", output_path / mode / corruptions[11], grayscale)
        Plotting.plot_and_save_img(image_variants[12], f"{mode}_{corruptions[12]}_{severity}", output_path / mode / corruptions[12], grayscale)
        Plotting.plot_and_save_img(image_variants[13], f"{mode}_{corruptions[13]}_{severity}", output_path / mode / corruptions[13], grayscale)

        # Save all images combined in one figure
        fig, axes = plt.subplots(nrows=3, ncols=5)

        # Plot the images
        axes[0, 0].imshow(X_orig, cmap=cmap)
        axes[0, 0].set_title('Original', loc='center')
        axes[0, 0].axis("off")

        axes[0, 1].imshow(image_variants[0], cmap=cmap)
        axes[0, 1].set_title("Gaussian\nNoise", loc='center')
        axes[0, 1].axis("off")

        axes[0, 2].imshow(image_variants[1], cmap=cmap)
        axes[0, 2].set_title("Shot\nNoise", loc='center')
        axes[0, 2].axis("off")

        axes[0, 3].imshow(image_variants[2], cmap=cmap)
        axes[0, 3].set_title("Impulse\nNoise", loc='center')
        axes[0, 3].axis("off")

        axes[0, 4].imshow(image_variants[3], cmap=cmap)
        axes[0, 4].set_title("Defocus\nBlur", loc='center')
        axes[0, 4].axis("off")

        axes[1, 0].imshow(image_variants[4], cmap=cmap)
        axes[1, 0].set_title("Motion\nBlur", loc='center')
        axes[1, 0].axis("off")

        axes[1, 1].imshow(image_variants[5], cmap=cmap)
        axes[1, 1].set_title("Zoom\nBlur", loc='center')
        axes[1, 1].axis("off")

        axes[1, 2].imshow(image_variants[6], cmap=cmap)
        axes[1, 2].set_title("Snow", loc='center')
        axes[1, 2].axis("off")

        axes[1, 3].imshow(image_variants[7], cmap=cmap)
        axes[1, 3].set_title("Frost", loc='center')
        axes[1, 3].axis("off")

        axes[1, 4].imshow(image_variants[8], cmap=cmap)
        axes[1, 4].set_title("Fog", loc='center')
        axes[1, 4].axis("off")

        axes[2, 0].imshow(image_variants[9], cmap=cmap)
        axes[2, 0].set_title("Brightness", loc='center')
        axes[2, 0].axis("off")

        axes[2, 1].imshow(image_variants[10], cmap=cmap)
        axes[2, 1].set_title("Contrast", loc='center')
        axes[2, 1].axis("off")

        axes[2, 2].imshow(image_variants[11], cmap=cmap)
        axes[2, 2].set_title("Elastic\nTransform", loc='center')
        axes[2, 2].axis("off")

        axes[2, 3].imshow(image_variants[12], cmap=cmap)
        axes[2, 3].set_title("Pixelate", loc='center')
        axes[2, 3].axis("off")

        axes[2, 4].imshow(image_variants[13], cmap=cmap)
        axes[2, 4].set_title("JPEG\nCompression", loc='center')
        axes[2, 4].axis("off")

        # Save and close the figure
        fig.tight_layout(h_pad=2)
        fig.savefig(output_path / f"{mode}" / f"{mode}_all_{severity}.png", bbox_inches='tight', pad_inches=0, dpi=300)
        fig.savefig(output_path / f"{mode}" / f"{mode}_all_{severity}.pdf", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
