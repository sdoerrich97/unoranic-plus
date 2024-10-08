"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Data Loader for the corruption revision objective.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
from pathlib import Path
import numpy as np
import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DataLoaderCustom:
    def __init__(self, dataset: str, data_path: Path, img_size: int, batch_size: int, corruption: str, seed: int):
        """
        Initialize the specified dataset as a torchvision dataset.

        :param dataset: Which dataset to load.
        :param data_path: Location of the data samples.
        :param img_size: The dimension the images should be padded to if necessary.
        :param batch_size: Batch size.
        :param corruption: What type of corruption should be applied to all data samples.
        :param seed: Random seed.
        """

        # Store the parameters
        self.dataset = dataset
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.corruption = corruption
        self.seed = seed
        self.num_workers = os.cpu_count()

        if self.corruption == 'PixelDropout':
            corrupt = A.PixelDropout(p=1)

        elif self.corruption == 'GaussianBlur':
            corrupt = A.GaussianBlur(p=1)

        elif self.corruption == 'ColorJitter':
            corrupt = A.ColorJitter(p=1)

        elif self.corruption == 'Downscale':
            corrupt = A.Downscale(scale_min=0.5, scale_max=0.9, p=1, interpolation=cv2.INTER_AREA)

        elif self.corruption == 'GaussNoise':
            corrupt = A.GaussNoise(p=1)

        elif self.corruption == 'InvertImg':
            corrupt = A.InvertImg(p=1)

        elif self.corruption == 'MotionBlur':
            corrupt = A.MotionBlur(p=1)

        elif self.corruption == 'MultiplicativeNoise':
            corrupt = A.MultiplicativeNoise(p=1)

        elif self.corruption == 'RandomBrightnessContrast':
            corrupt = A.RandomBrightnessContrast(p=1)

        elif self.corruption == 'RandomGamma':
            corrupt = A.RandomGamma(p=1)

        elif self.corruption == 'Solarize':
            corrupt = A.Solarize(threshold=128, p=1)

        else:
            corrupt = A.Sharpen(p=1)

        self.transform = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_AREA, p=1.0),
            corrupt,
            A.ToFloat(),
            ToTensorV2()
        ])

        self.transform_prime = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_AREA, p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        # Load the dataset
        self.train_set = MEDMNIST(self.data_path, self.transform_prime, self.transform, corruption, self.seed, "train")
        self.val_set = MEDMNIST(self.data_path, self.transform_prime, self.transform, corruption, self.seed, "val")
        self.test_set = MEDMNIST(self.data_path, self.transform_prime, self.transform, corruption, self.seed, "test")

        # Create the data loader
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=False)
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
    def __init__(self, data_path: Path, transform_prime: A.Compose, transform: A.Compose, corruption: str, seed: int,
                 samples_set: str):
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
        :param corruption: What type of corruption should be applied to all data samples
        :param seed: Random seed.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
        self.transform_prime = transform_prime
        self.transform = transform
        self.corruption = corruption
        self.seed = seed
        self.samples_set = samples_set

        # Get all samples for the current set
        self.samples = np.load(str(self.data_path))[f"{samples_set}_images"]
        self.labels = np.load(str(self.data_path))[f"{samples_set}_labels"]

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.samples)

    def __getitem__(self, index: int):
        """Load and return a sample together with distorted variants of that sample."""

        # Extract the samples path
        sample_name = f"{index}"
        sample_orig, label_orig = self.samples[index], self.labels[index]

        # Apply the corruption to the original sample
        random.seed(self.seed)
        sample = self.transform(image=sample_orig)["image"]
        random.seed(self.seed)
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the loaded sample(s)
        return sample_orig, label_orig, sample, sample_name
