#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def get_input_args():

    # Create Parse using Argument Parser
    parser = argparse.ArgumentParser()

    # Create the required command line arguments:
    # 1. directory to save checkpoints
    # 2. architecture
    # 3. hyperparameters: learning rate, hidden_units, epochs
    # 4. use gpu for training

    parser.add_argument('data_dir', type = str, default = 'flowers', help = 'data directory required')

    parser.add_argument('--save_dir', type = str, default = 
        'checkpoints', help = 'path to save checkpoint')
    
    parser.add_argument('--arch', type = str, default = 
        'resnet', help = 'CNN Model Architecture to use')
    
    parser.add_argument('--learning_rate', type = float, default = 
        0.001, help = 'training learning rate')
    
    parser.add_argument('--hidden_units', type = int, default = 
        1024, help = 'hidden units for training')
    
    parser.add_argument('--epochs', type = int, default = 
        14, help = 'number of epochs for training')
    
    # https://stackoverflow.com/questions/5262702/argparse-module-how-to-add-option-without-any-argument
    parser.add_argument('-g','--gpu', action ='store_true')

    return parser.parse_args()

def get_input_args_predict():
    # Create Parse using Argument Parser
    parser = argparse.ArgumentParser()

    # Create the required command line arguments:
    # 1. path to image
    # 2. checkpoint location
    # 3. top K most likely classes
    # 4. mapping of categories to real names
    # 5. use gpu for inference

    parser.add_argument('img_path', type = str, help = 'image path required')

    parser.add_argument('checkpoint', type = str, help = 'checkpoint path required')

    parser.add_argument('--top_k', type = int, default = 
        1, help = 'Top K most likely classes')

    parser.add_argument('--category_names', type = str, default = 
        'cat_to_name.json', help = 'File for mapping of categories to real names')

    parser.add_argument('-g','--gpu', action ='store_true')

    return parser.parse_args()