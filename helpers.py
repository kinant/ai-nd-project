#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Kinan Turman
# DATE CREATED: Feb. 23, 2019                       
# PURPOSE: Helper functions, one for opening a .json file for class id to name mapping.
# The other helper function is the image process function, which processes
# an image into a PyTorch tensor.

import numpy as np
import torch

import json
from math import floor
from PIL import Image

def open_label_mapping_file(filename):
    """
        This function loads in a mapping from category label to category name

        Parameters:
        - filename: the .json file to be read with the mapping

        Returns:
        - cat_to_name: dictionary mapping the integer encoded categories to the actual names of the flowers.
    """
    cat_to_name = None
    
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    # open the image
    image = Image.open(image_path)
    
    # print("Original Image size: ", image.size)
    
    # first resize the images where the shortest side is 256 px
    width, height = image.size
    size = 256, 256
    
    newwidth, newheight = None, None
    
    # if the height is the shorter side
    if height < width:
        # find ratio between larger and smaller side
        ratio = float(width) / float(height)
        # resize smaller side to 256
        newheight = 256
        # resize larger side to 256 * ratio
        newwidth = int(floor(ratio * size[1]))  
    # else, the width is the shorter side
    else:
        # find ratio between larger and smaller side
        ratio = float(height)/float(width)
        # resize smaller side to 256
        newwidth = 256
        # resize larger side to 256 * ratio
        newheight = int(floor(ratio * size[0]))
    
    
    # print("W: {}, H: {}".format(newwidth, newheight))
    
    # resize the image
    image = image.resize((newwidth, newheight), Image.ANTIALIAS)

    # print("Resized Image (keep aspect ratio): ", image.size)
    
    # perform center crop
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width, height = image.size   # Get dimensions
    new_height, new_width = 224, 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom))
    # print("cropped image size: ", image.size)
    
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