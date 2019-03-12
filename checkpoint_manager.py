#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Kinan Turman
# DATE CREATED: Feb. 23, 2019                       
# PURPOSE: A basic manager for saving and loading checkpoints. Used by train.py to save
# a checkpoint and by predict.py to load a checkpoint/model to predict the classes for 
# a given image.

import torch
import build_model as bm
import torchvision
from torchvision import datasets, models, utils

import os

def save_model(model, num_classes, class_to_idx_mapping, c_arch, hidden_units,
    learning_rate, dropout, path):

    """
        This function saves the checkpoint

        Parameters:
        - model: the model we are saving (using state_dict())
        - num_classes: the output size
        - class_to_idx_mapping: the mapping of classes to indices which we get from the test set
        - c_arch: the choice of architecture
        - hidden_units: hidden units
        - learning_rate: the learning rate
        - dropout: the dropout
        - path: where we want to save the checkpoint

        Returns:
        - None
    """

    model.class_to_idx = class_to_idx_mapping
    m_state_dict = model.state_dict()

    checkpoint = {'output_size': num_classes,
                  'dropout' : dropout,
                  'choice_model': c_arch,
                  'hidden_unit' : hidden_units,
                  'learning_rate': learning_rate,
                  'model_state_dict': m_state_dict,
                  'class_to_idx': model.class_to_idx}
    
    if path:
        # create the directory (first check that it does not exist):
        if not os.path.exists(path):
            os.mkdir(path)

        torch.save(checkpoint, path + '/' + 'checkpoint.pth')
    else:
        torch.save(checkpoint, 'checkpoint.pth')

    print('Model saved!')

def load_checkpoint(path):
    """
        This loads a checkpoint

        Parameters:
        - path: path to the checkpoint (dir + filename)

        Returns:
        - None
    """

    # load checkpoint
    checkpoint = torch.load(path)
    # load parameters
    num_classes = checkpoint['output_size']
    c_arch = checkpoint['choice_model']
    dropout = checkpoint['dropout']
    hidden_unit = checkpoint['hidden_unit']
    # load the model
    model = bm.initialize_model(c_arch, num_classes, hidden_unit, dropout)
    # load model state and class idx
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']  
    
    # print("Model Loaded: ", model)
    # print()
    #print("Class to idx: ", model.class_to_idx)

    return model