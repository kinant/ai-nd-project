#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from get_input_args import get_input_args
import data_loader as dl
import build_model as bm
import checkpoint_manager as cm

import numpy as np
import torch
import time
import copy
import json

import random
from PIL import Image

def main():

    # command line arguments
    c_arch = None
    c_num_epochs = None
    c_learning_rate = None
    c_hidden_units = None
    c_use_gpu = False
    c_save_dir = None

    # other parameters
    dropout = 0.4
    batch_size = 64
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_px = 224
    num_classes = None

    in_args = get_input_args()

    c_data_dir = in_args.data_dir
    c_save_dir = in_args.save_dir
    c_arch = in_args.arch
    c_learning_rate = in_args.learning_rate
    c_num_epochs = in_args.epochs
    c_hidden_units = in_args.hidden_units
    c_use_gpu = in_args.gpu

    print("Data Dir: {}, Save dir: {}, Arch: {}, LR: {}, Epochs: {}, H Units: {}, GPU: {}".format(c_data_dir, c_save_dir, c_arch,
        c_learning_rate, c_num_epochs, c_hidden_units, c_use_gpu))

    # get the data transforms:
    image_datasets, dataloaders = dl.load_data(c_data_dir, img_px, mean, std, batch_size)

    # print("Data transforms: ", data_transforms)
    # print("Image_datasets: ", image_datasets)
    # print("dataloaders: ", dataloaders)

    # Initialize the model:
    _, _, num_classes = dl.get_data_info(image_datasets)

    model_ft = bm.initialize_model(c_arch, num_classes, c_hidden_units, dropout)

    # print("Model: ", model_ft)

    device = torch.device("cuda:0" if torch.cuda.is_available() and c_use_gpu else "cpu")
    # print("Device: ", device)

    # get criterion and optimizer
    criterion, optimizer = bm.setup_criterion_optim(model_ft, device, c_learning_rate)

    # print("Criterion: ", criterion)
    # print("Optimizer: ", optimizer)

    # Now we can train and evaluate
    model_ft = train_model(c_arch, model_ft, device, dataloaders, criterion, optimizer, num_epochs=c_num_epochs)

    # Check accuracy on test data
    # check_accuracy_on_test(model_ft, device, dataloaders['test'])

    # Save the model
    cm.save_model(model_ft, num_classes, image_datasets['train'].class_to_idx, c_arch, 
        c_hidden_units, c_learning_rate, dropout, c_save_dir)

def train_model(arch, model, device, dataloaders, criterion, optimizer, num_epochs=25):
    
    print("Starting training with: ", arch)
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def check_accuracy_on_test(model, device, testloader):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images:{}%'.format(100 * correct / total))

# Call to main function to run the program
if __name__ == "__main__":
    main()