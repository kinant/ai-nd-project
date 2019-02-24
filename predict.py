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
    c_mapping = None
    c_use_gpu = False

    in_args = get_input_args_predict()

    # Get command line arguments
    c_img_path = in_args.img_path
    c_chk_path = in_args.checkpoint
    c_top_k = in_args.top_k
    c_mapping = in_args.category_names
    c_use_gpu = in_args.gpu

    print("Image Path: {}, Checkpoint: {}, Top K: {}, Mapping: {}, GPU: {}".format(c_img_path, c_chk_path, c_top_k,
        c_mapping, c_use_gpu))

    model = checkpoint_manager.load_checkpoint(c_chk_path)
    probs, classes = predict(c_img_path, model, c_use_gpu, c_top_k)

    print()
    print("Probs: ", probs)
    print("Classes: ", classes)


def predict(image_path, model, use_gpu, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # set the mode for inference
    model.eval()

    # use gpu if required
    if use_gpu:
        model.to('cuda:0')

    # process the image
    image = hlp.process_image(image_path)
    image = np.expand_dims(image, 0)
    
    img_to_fwd = torch.from_numpy(image)
    
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img_to_fwd)

    ps = torch.exp(output)
    
    # print("PS: ", ps)
    
    probs, classes = ps.topk(5)
    
    # probs and classes are tensors, so we convert to lists so we return
    # as is required
    top_probs = probs.detach().numpy().tolist()[0]
    top_classes = classes.detach().numpy().tolist()[0]
    
    # top classes are not strings, so we just convert
    top_classes = [str(val) for val in top_classes]
    
    # print("Probs: ", top_probs)
    # print("Classes: ", top_classes)
    return top_probs, top_classes

    # Call to main function to run the program
if __name__ == "__main__":
    main()
