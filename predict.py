#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision
from torchvision import models, utils

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

    in_args = get_input_args_predict()

    # Get command line arguments
    c_img_path = in_args.img_path
    c_chk_path = in_args.checkpoint
    c_top_k = in_args.top_k
    c_mapping_path = in_args.category_names
    c_use_gpu = in_args.gpu

    #print("Image Path: {}, Checkpoint: {}, Top K: {}, Mapping: {}, GPU: {}".format(c_img_path, c_chk_path, c_top_k,
    #    c_mapping_path, c_use_gpu))

    model = checkpoint_manager.load_checkpoint(c_chk_path)

    device = torch.device("cuda" if torch.cuda.is_available() and c_use_gpu else "cpu")

    probs, classes = predict(c_img_path, model, device, c_top_k)

    if c_mapping_path is not None:
        cat_to_name = hlp.open_label_mapping_file(c_mapping_path)
        classes = map_cat_to_real_names(model, classes, cat_to_name)

    print_results(probs, classes)

def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # set the mode for inference
    model.eval()
    model.to(device)
    
    # process the image
    image = hlp.process_image(image_path);
    image = np.expand_dims(image, 0)
    
    img_to_fwd = torch.from_numpy(image)
    img_to_fwd = img_to_fwd.to(device)
    
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img_to_fwd)

    ps = torch.exp(output)
    
    probs, classes = ps.topk(topk)
    
    # probs and classes are tensors, so we convert to lists so we return
    # as is required
    top_probs = probs.cpu().detach().numpy().tolist()[0]
    top_classes = classes.cpu().detach().numpy().tolist()[0]
    
    # https://knowledge.udacity.com/questions/31597
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    classes = []
    
    for cls in top_classes:
        c = idx_to_class[cls]
        classes.append(c)
    
    # top classes are not strings, so we just convert
    return top_probs, classes

    # Call to main function to run the program

def map_cat_to_real_names(model, classes, cat_to_name):
    labels = []
    
    for cls in classes:
        labels.append(cat_to_name[cls])
    
    return labels

def print_results(probs, classes):
    print()
    print("Prediction Results: ")
    print("=================================")
    
    for i in range(len(probs)):
        print("Class: {}, Probability: {}".format(classes[i], probs[i]))

if __name__ == "__main__":
    main()
