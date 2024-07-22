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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from imagecorruptions import corrupt
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DataLoaderCustom:
    def __init__(self, dataset: str, data_path: Path, img_size: int, in_channel: int, batch_size: int, corruption: str, severity: int):
        """
        Initialize the specified dataset as a torchvision dataset.

        :param dataset: Which dataset to load.
        :param data_path: Location of the data samples.
        :param img_size: The dimension the images should be padded to if necessary.
        :param in_channel: Channel dimension of input.
        :param batch_size: Batch size.
        :param corruption: What type of corruption should be applied to all data samples.
        :param severity: How strong should the corruption be.
        """

        # Store the parameters
        self.dataset = dataset
        self.data_path = data_path
        self.img_size = img_size
        self.in_channel = in_channel
        self.batch_size = batch_size
        self.corruption = corruption
        self.severity = severity
        self.num_workers = os.cpu_count()

        # Create the augmentations
        self.transform_pre = A.Compose([
            A.Resize(height=32, width=32, interpolation=cv2.INTER_AREA, p=1.0)
        ])

        self.transform_post = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_AREA, p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        # Load the dataset
        self.train_set = MEDMNIST(self.data_path, self.in_channel, self.transform_pre, self.transform_post, self.corruption, self.severity, "train")
        self.val_set = MEDMNIST(self.data_path, self.in_channel, self.transform_pre, self.transform_post, self.corruption, self.severity, "val")
        self.test_set = MEDMNIST(self.data_path, self.in_channel, self.transform_pre, self.transform_post, self.corruption, self.severity, "test")

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
    def __init__(self, data_path: Path, in_channel: int, transform_pre: A.Compose, transform_post: A.Compose,
                 corruption: str, severity: int, samples_set: str):
        """
        Initialize a corrupted MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the given dataset as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param in_channel: Channel dimension of input.
        :param transform_pre: Loading pipeline for the original image.
        :param transform_post: Loading pipeline for the input image.
        :param corruption: What type of corruption should be applied to all data samples
        :param severity: How strong should the corruption be.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
        self.in_channel = in_channel
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.corruption = corruption
        self.severity = severity
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

        # Apply the preprocessing to the original sample
        sample_pre = self.transform_pre(image=sample_orig)["image"]

        # Apply the corruption
        corrupted = corrupt(sample_pre, corruption_name=self.corruption, severity=self.severity)

        # Apply the postprocessing to the original sample
        sample = self.transform_post(image=corrupted)["image"]
        sample_orig = self.transform_post(image=sample_orig)["image"]

        # Check wether the input image had only one channel
        if self.in_channel == 1:
            sample = torch.mean(sample, dim=0, keepdim=True)

        # Return the loaded sample(s)
        return sample_orig, label_orig, sample, sample_name
