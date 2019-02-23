#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models, utils
from torch.autograd import Variable
from torch.optim import lr_scheduler

import time
import os
import copy
from collections import OrderedDict
import json

import random
from PIL import Image

def setup_criterion_optim(model_ft, device, learning_rate):
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    
    return criterion, optimizer

def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, hidden_units, c_drop, feature_extract=True, use_pretrained=True):

    model_ft = None

    if model_name == "resnet":

        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features
        
        model_ft.fc =  nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units)),
                          ('drop', nn.Dropout(c_drop)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    elif model_name == "alexnet":

        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        
        model_ft.classifier[6] = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units)),
                          ('drop', nn.Dropout(c_drop)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    elif model_name == "vgg":

        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        
        model_ft.classifier[6] = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units)),
                          ('drop', nn.Dropout(c_drop)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    elif model_name == "densenet":

        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
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