#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models, utils

from collections import OrderedDict

def setup_criterion_optim(model_ft, device, learning_rate):    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.classifier.parameters(), lr=learning_rate)
    
    return criterion, optimizer

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def initialize_model(model_name, num_classes, hidden_units, c_drop):

    model_ft = None

    if model_name == "alexnet":

        model_ft = models.alexnet(pretrained=True)
        
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier[1].in_features
        
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
        num_ftrs = model_ft.classifier[0].in_features
        
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
        num_ftrs = model_ft.classifier.in_features
        
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