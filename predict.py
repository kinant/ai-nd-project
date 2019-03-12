#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Kinan Turman
# DATE CREATED: Feb. 23, 2019                       
# PURPOSE: Predict flower name from an image with predict.py along with the probability of that name. 
# Can also predict the Top K classes and use category name mapping to show the real names. 

import numpy as np
import torch
import torchvision
from torchvision import models, utils
import torch.nn.functional as F

import helpers as hlp
import checkpoint_manager

from get_input_args import get_input_args_predict

def main():
    
    # command line arguments
    c_img_path = None
    c_chk_path = None
    c_top_k = None
    c_mapping_path = None
    c_use_gpu = False

    # Get command line arguments
    in_args = get_input_args_predict()

    c_img_path = in_args.img_path
    c_chk_path = in_args.checkpoint
    c_top_k = in_args.top_k
    c_mapping_path = in_args.category_names
    c_use_gpu = in_args.gpu

    print("Running predict.py with the following arguments: ")
    print("Image Path: {}\nCheckpoint: {}\nTop K: {}\nMapping: {}\nGPU: {}".format(c_img_path, c_chk_path, c_top_k,
        c_mapping_path, c_use_gpu))

    # load the checkpoint
    model = checkpoint_manager.load_checkpoint(c_chk_path)

    # set the device (gpu or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() and c_use_gpu else "cpu")

    # call the predict function and get the topK (c_top_k) classes and probabilities 
    probs, classes = predict(c_img_path, model, device, c_top_k)

    # check to see if we want to map the classes to the category names (one of the command
    # line arguments)
    if c_mapping_path is not None:
        cat_to_name = hlp.open_label_mapping_file(c_mapping_path)
        classes = map_cat_to_real_names(classes, cat_to_name)

    # print the results
    print_results(probs, classes)

def predict(image_path, model, device, topk):
    """
        Predict the class (or classes) of an image using a trained deep learning model.

        Parameters:
        - image_path: path to the image for which we will predict the class(es)
        - model: the model to be used
        - device to be used: gpu or cpu
        - topk: the number of K most likely classes we want to calculate/return

        Returns:
        - top_probs: the top probabilities
        - classes: the top classes
    """
    # set the mode for inference
    model.eval()
    
    # set the device
    model.to(device)
    
    # process the image
    image = hlp.process_image(image_path)
    image = np.expand_dims(image, 0)
    
    img_to_fwd = torch.from_numpy(image)
    img_to_fwd = img_to_fwd.to(device)
    
    # Turn off gradients to speed up this part
    with torch.no_grad():
        # fwd pass get logits
        output = model.forward(img_to_fwd)

    # Calculate the class probabilities for img
    # ps = torch.exp(output)
    # Calculate the class probabilities (softmax) for img
    ps = F.softmax(output, dim=1)
    # get the top K largest values
    probs, classes = ps.topk(5)
    
    # probs and classes are tensors, so we convert to lists so we return
    # as is required
    top_probs = probs.cpu().detach().numpy().tolist()[0]
    top_classes = classes.cpu().detach().numpy().tolist()[0]
    
    # I was getting the wrong class labels when converting,
    # the solution in the following helped me:
    # https://knowledge.udacity.com/questions/31597
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    classes = []
    
    # convert the classes using idx_to_class
    for cls in top_classes:
        c = idx_to_class[cls]
        classes.append(c)
    
    # return the 
    return top_probs, classes

def map_cat_to_real_names(classes, cat_to_name):
    """
        Maps class categories to real names

        Parameters:
        - classes: the classes (list of ids)
        - cat_to_name: dictionary mapping the integer encoded categories to the actual names of the flowers

        Returns:
        - labels: the classes mapped to their actual names
    """
    labels = []
    
    for cls in classes:
        labels.append(cat_to_name[cls])
    
    return labels

def print_results(probs, classes):
    """
        Prints the results of predict.py

        Parameters:
        - probs: the probabilities to print
        - classes: the classes to print

        Returns:
        - None
    """
    print()
    print("Prediction Results: ")
    print("=================================")
    
    for i in range(len(probs)):
        print("Class: {}, Probability: {}".format(classes[i], probs[i]))

if __name__ == "__main__":
    main()
