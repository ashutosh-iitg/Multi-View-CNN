import json
import logging
import os
import shutil

import torch
from collections import defaultdict

def save_dict_to_json(dict, json_path):
    """Saves dict of floats in json file
    Args:
        dict: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        json.dump(dict, f, indent=4)
        
def load_json(json_path):
    """Reads a json file from given path
    Args:
        json_path: (string) path to json file
    """
    with open(json_path) as f:
        json_dict = json.load(f)
    
    return json_dict

def load_model(model_path, model):
    """Loads model parameters (state_dict) from model_path for inference.
    Args:
        model_path: (string) path to model state_dict
        model: (torch.nn.Module) model for which the parameters are loaded
    """
    if not os.path.exists(model_path):
        raise("File doesn't exist {}".format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['state_dict'])