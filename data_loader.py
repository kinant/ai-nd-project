#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torchvision import datasets, transforms, utils
import os

def get_data_info(image_datasets):
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    return dataset_sizes, class_names, num_classes

def load_data(data_dir, img_px, mean, std, batch_size):
    
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders