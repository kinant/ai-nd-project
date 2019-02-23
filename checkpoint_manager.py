import torch
import build_model as bm
import torchvision
from torchvision import datasets, models, utils

import os

def save_model(model, num_classes, dataset_idx, c_arch, hidden_units,
    learning_rate, dropout, path):
    #model.class_to_idx = image_datasets['train'].class_to_idx
    model.class_to_idx = dataset_idx
    m_state_dict = model.state_dict()

    checkpoint = {'output_size': num_classes,
                  'dropout' : dropout,
                  'choice_model': c_arch,
                  'hidden_unit' : hidden_units,
                  'learning_rate': learning_rate,
                  'model_state_dict': m_state_dict,
                  'class_to_idx': model.class_to_idx}
    
    # create the directory:
    os.mkdir(path)

    torch.save(checkpoint, path + '/' + 'checkpoint.pth')
    print('Model saved!')

def load_checkpoint(path):
    # load parameters
    checkpoint = torch.load(path)
    num_classes = checkpoint['output_size']
    c_arch = checkpoint['choice_model']
    dropout = checkpoint['dropout']
    hidden_unit = checkpoint['hidden_unit']
    # load the model
    model = bm.initialize_model(c_arch, num_classes, hidden_unit, dropout)
    # load model state and class idx
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_idx = checkpoint['class_to_idx']  
    
    print("Model Loaded: ", model)

    return model