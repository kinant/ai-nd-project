#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Kinan Turman
# DATE CREATED: Feb. 23, 2019                       
# PURPOSE: Builds a model from given hyper-parameters. Also builds the criterion and optimizer. These are to
# be used in train.py

import numpy as np

import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models, utils

from collections import OrderedDict

def setup_criterion_optim(model_ft, learning_rate):    
    """
        This function just sets up the criterion and optimizer for our nework training

        Parameters:
        - model_ft: the model we are using/built
        - learning_rate: the learning rate

        Returns:
        - criterion: the criterion that will be used for training
        - optimizer: the optimizer that will be used for training
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.classifier.parameters(), lr=learning_rate)
    
    return criterion, optimizer

def set_parameter_requires_grad(model):
    """
        This function is just a simpler way to set the .requires_grad attriute of the 
        parameters of a model to false. Modified version from tutorial at:
        https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#set-model-parameters-requires-grad-attribute

        Parameters:
        - model: the model we are building

        Returns:
        - None
    """

    for param in model.parameters():
        param.requires_grad = False

def initialize_model(model_name, num_classes, hidden_units, c_drop):
    """
        Initializes a model for training. Basically takes a pre-trained
        model and replaces the classifier. For this function, I used what
        we learned in class, as well as the following two tutorials from PyTorch.org:
        - https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
        - https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data

        Parameters:
        - model_name: the name of the model architecture we want to use, ie. "vgg"
        - num_classes: the number of classes, in other words, the output_size
        - hidden_units: the number of hidden units, to be used in classifier
        - c_drop: our choice of dropout for the classifier

        Returns:
        - model_ft: our modified (classifier replaced) model 
    """

    model_ft = None
    # use if and elif statements to check which one of the pre-trained
    # models we want to use

    if model_name == "alexnet":

        model_ft = models.alexnet(pretrained=True)
        
        set_parameter_requires_grad(model_ft)

        # get the number of input features from the 1st layer
        # of the 'original' classifier
        num_ftrs = model_ft.classifier[1].in_features
        
        # replace the entire classifier
        model_ft.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units)),
                          ('drop', nn.Dropout(c_drop)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    elif model_name == "vgg":

        model_ft = models.vgg19(pretrained=True)
        set_parameter_requires_grad(model_ft)
        
        # get the number of input features from the 0th layer
        # of the 'original' classifier
        num_ftrs = model_ft.classifier[0].in_features
        
        # replace the entire classifier
        model_ft.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units)),
                          ('drop', nn.Dropout(c_drop)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

   
    elif model_name == "densenet":


        model_ft = models.densenet121(pretrained=True)
        set_parameter_requires_grad(model_ft)
        
        # get the number of input features from
        # of the 'original' classifier 
        num_ftrs = model_ft.classifier.in_features
        
        # replace the entire classifier
        model_ft.classifier =  nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units)),
                          ('drop', nn.Dropout(c_drop)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
            
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft