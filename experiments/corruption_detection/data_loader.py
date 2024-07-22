"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Data Loader for a corruption detection objective.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
from pathlib import Path
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DataLoaderCustom:
    def __init__(self, dataset: str, data_path: Path, img_size: int, batch_size: int, seed: int, train: bool):
        """
        Initialize the specified dataset as a torchvision dataset.

        :param dataset: Which dataset to load.
        :param data_path: Location of the data samples.
        :param img_size: The dimension the images should be padded to if necessary.
        :param batch_size: Batch size.
        :param seed: Seed for random operations.
        :param train: whether the data is loaded for training or for inference.
        """

        # Store the parameters
        self.dataset = dataset
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        self.train = train
        self.num_workers = os.cpu_count()

        # Create the augmentations
        self.transform = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_AREA, p=1.0),
            A.OneOf([
                A.PixelDropout(p=1),
                A.GaussianBlur(p=1),
                A.ColorJitter(p=1),
                A.Downscale(scale_min=0.5, scale_max=0.9, p=1, interpolation=cv2.INTER_AREA),
                A.GaussNoise(p=1),
                A.InvertImg(p=1),
                A.MotionBlur(p=1),
                A.MultiplicativeNoise(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.Solarize(threshold=128, p=1),
                A.Sharpen(p=1),
            ], p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        self.transform_prime = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_AREA, p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        if self.train:
            shuffle = True

        else:
            shuffle = False

        # Load the dataset
        self.train_set = MEDMNIST(self.data_path, self.transform_prime, self.transform, self.seed, "train")
        self.val_set = MEDMNIST(self.data_path, self.transform_prime, self.transform, self.seed, "val")
        self.test_set = MEDMNIST(self.data_path, self.transform_prime, self.transform, self.seed, "test")

        # Create the data loader
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.val_loader = DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False)

    def get_train_loader(self):
        """Get the train loader."""

        return self.train_loader

    def get_val_loader(self):
        """Get the validation loader."""

        return self.val_loader

    def get_test_loader(self):
        """Get the test loader."""

        return self.test_loader


class MEDMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, seed: int, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the given dataset as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param seed: Seed for random operations.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
        self.transform_prime = transform_prime
        self.transform = transform
        self.seed = seed
        self.samples_set = samples_set

        # Get all samples for the current set
        self.samples = np.load(str(self.data_path))[f"{samples_set}_images"]
        self.labels = np.load(str(self.data_path))[f"{samples_set}_labels"]

        # Set the random seed
        random.seed(self.seed)

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Load and return a sample together with its label, a potentially distorted variant and a label for the distortion
        """

        # Extract the samples path
        sample_name = f"{index}"
        sample_orig, label_orig = self.samples[index], self.labels[index]

        # Apply the corruption to the original sample
        if random.random() < 0.5:
            sample = self.transform(image=sample_orig)["image"]
            label = 1

        else:
            sample = self.transform_prime(image=sample_orig)["image"]
            label = 0

        # Covert the original sample to a tensor
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the loaded sample(s)
        return sample_orig, label_orig, sample, np.array([label]), sample_name
