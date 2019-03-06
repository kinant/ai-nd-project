#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

import json
from PIL import Image

def open_label_mapping_file(filename):
    cat_to_name = None
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    # open the image
    image = Image.open(image_path)
    
    # Resize the image
    image = image.resize((256, 256))

    # perform center crop
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width, height = image.size   # Get dimensions
    # print("image size 1: ", image.size)
    new_height, new_width = 224, 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom))
    # print("image size 2: ", image.size)
    
    # convert encoded color channels and convert to floats (divide by 255)
    np_image = np.array(image) / 255
    # print(np_image)
    
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean) / std
    
    # finally, transpose
    # print("shape 1: ", np_image.shape)
    np_image = np_image.transpose((2, 0, 1))
    # print("transposed shape: ", np_image.shape)
    
    # Originally, I was returning a numpy array, as I thought these were the instructions, but
    # when trying to test, it would not work. 
    # Found solution at: https://knowledge.udacity.com/questions/29173
    # We have to convert to a tensor before we return it
    return torch.Tensor(np_image)