#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Kinan Turman
# DATE CREATED: Feb. 23, 2019                       
# PURPOSE: Loads the data for our neural network. Also provides relevant information.
# Used in train.py

import numpy as np
import torch
from torchvision import datasets, transforms, utils
import os

def get_data_info(image_datasets):
    """
        This function just returns information about our image datasets

        Parameters:
        - image_datasets: the image_datasets for the proejct

        Returns:
        - dataset_sizes: the size for each dataset
        - class_names: the name for each class
        - num_classes: the number of classes, which is the output size (this is the only value we really use)
    """
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    return dataset_sizes, class_names, num_classes

def load_data(data_dir, img_px, mean, std, batch_size):
    """
        This function loads the data for our project

        Parameters:
        - data_dir: directory of data (ie, 'flowers')
        - img_px: standardized image pixel size
        - mean: the mean for standardization
        - std: the standard deviation for standardization
        - batch_size: desired batch size

        Returns:
        - image_datasets: our image data sets for project
        - dataloaders: dataloaders for the project
    """

    # directories for data sets
    # train_dir = data_dir + '/train'
    # valid_dir = data_dir + '/valid'
    # test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_px),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(img_px + 1),
            transforms.CenterCrop(img_px),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_px + 1),
            transforms.CenterCrop(img_px),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # Used modified code from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # to create the image datasets and the dataloaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders