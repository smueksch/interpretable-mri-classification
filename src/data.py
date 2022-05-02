import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split


def get_img_dataset(data_dir: str, logger: logging.Logger, transform=None):
    # Define data transform
    train_transform = []
    if transform is not None:
        train_transform += transform
    train_transform += [
        transforms.Resize(128),  # resize shortest side to 128 pixels
        transforms.CenterCrop(
            128
        ),  # crop longest side to 128 pixels at center
        transforms.ToTensor(),  # convert PIL image to tensor
    ]
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(
        [
            transforms.Resize(128),  # resize shortest side to 128 pixels
            transforms.CenterCrop(
                128
            ),  # crop longest side to 128 pixels at center
            transforms.ToTensor(),  # convert PIL image to tensor
        ]
    )

    # Initialize train/test sets
    data_path = Path(data_dir)
    train_dataset = ImageFolder(data_path, transform=train_transform)
    test_dataset = ImageFolder(data_path, transform=test_transform)
    classes = train_dataset.find_classes(data_path)[1]
    logger.info(
        "Loaded samples into dataset with label "
        + f"'no'={classes['no']} and 'yes'={classes['yes']}"
    )

    # Split dataset into train/test sets and stratify over labels to balance
    # datasets with set seed
    # DO NOT CHANGE THE SEED
    generator = torch.Generator().manual_seed(390397)
    train_len = int(0.8 * len(train_dataset))
    test_len = int((len(train_dataset) - train_len) / 2)
    train_dataset = random_split(
        dataset=train_dataset,
        lengths=[train_len, test_len, test_len],
        generator=generator,
    )[0]
    val_dataset, test_dataset = random_split(
        dataset=test_dataset,
        lengths=[train_len, test_len, test_len],
        generator=generator,
    )[1:]

    return train_dataset, val_dataset, test_dataset


def get_radiomics_dataset(data_dir: str):
    # Define relevant features
    rel_feat = [
        "diagnostics_Versions_PyRadiomics",
        "diagnostics_Versions_Numpy",
        "diagnostics_Versions_SimpleITK",
        "diagnostics_Versions_PyWavelet",
        "diagnostics_Versions_Python",
        "diagnostics_Configuration_Settings",
        "diagnostics_Configuration_EnabledImageTypes",
        "diagnostics_Image-original_Hash",
        "diagnostics_Image-original_Dimensionality",
        "diagnostics_Image-original_Spacing",
        "diagnostics_Image-original_Size",
        "diagnostics_Image-original_Mean",
        "diagnostics_Image-original_Minimum",
        "diagnostics_Image-original_Maximum",
        "diagnostics_Mask-original_Hash",
        "diagnostics_Mask-original_Spacing",
        "diagnostics_Mask-original_Size",
        "diagnostics_Mask-original_BoundingBox",
        "diagnostics_Mask-original_VoxelNum",
        "diagnostics_Mask-original_VolumeNum",
        "diagnostics_Mask-original_CenterOfMassIndex",
        "diagnostics_Mask-original_CenterOfMass",
    ]

    # Load train/test sets from csvs
    train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    train_data.drop(inplace=True, axis=1, labels=rel_feat)
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    val_data = pd.read_csv(os.path.join(data_dir, "validation_data.csv"))
    val_data.drop(inplace=True, axis=1, labels=rel_feat)
    val_labels = np.load(os.path.join(data_dir, "validation_labels.npy"))
    test_data = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
    test_data.drop(inplace=True, axis=1, labels=rel_feat)
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    return (
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
    )