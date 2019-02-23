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
    
    parser.add_argument('--gpu', type = bool, default = 
        True, help = 'use gpu for training')

    return parser.parse_args()