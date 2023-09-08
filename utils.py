import os
import gc
import json
import pickle
import yaml
from PIL import Image
import subprocess
import pandas as pd
from typing import Optional
import torch
from torch.optim import AdamW

def load_from_yaml(path_to_file: str):
    print(f"loading data from .yaml file @ {path_to_file}")
    with open(path_to_file) as file:
        _dict = yaml.safe_load(file)
    return _dict

path_to_config_file = "./config.yaml"
config = load_from_yaml(path_to_file=path_to_config_file)

def load_from_txt(path_to_file: str):
    print(f"loading data from .txt file @ {path_to_file}")
    with open (path_to_file, "r") as myfile:
        data = myfile.read().splitlines()
    return data

def save_to_json(path_to_file: str, data: list):
    with open(path_to_file, 'w') as outfile:
        json.dump(data, outfile)
    print(f"file saved @ loc: {path_to_file}")

def load_from_json(path_to_file: str):
    # print(f"loading data from .json file @ {path_to_file}")
    with open(path_to_file, "r") as json_file:
        _dict = json.load(json_file)
    return _dict

def save_to_pickle(data_list, path_to_file):
    with open(path_to_file, 'wb') as file:
        pickle.dump(data_list, file)
    print(f"file saved @ loc: {path_to_file}")

def load_from_pickle(path_to_file):
    print(f"loading data from .pkl file @ {path_to_file}")
    with open(path_to_file, 'rb') as file:
        data_list = pickle.load(file)
    return data_list

def load_image(path_to_file):
    # print(f"loading image from .jpeg file @ {path_to_file}")
    image = Image.open(path_to_file).convert("RGB").resize((400, 400))
    return image

def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')
        gpu_memory_usage = [int(memory_used) for memory_used in output.strip().split('\n')]
        return gpu_memory_usage
    except Exception as e:
        print(f"Error while running nvidia-smi: {e}")
        return None
    
def pad_seq(tensor: torch.tensor,
            dim: int,
            max_len: int,
            pad_token_id: int=0):
    if max_len > tensor.shape[0]:
        return torch.cat([tensor, torch.ones(max_len - tensor.shape[0], dim) * pad_token_id])
    else:
        return tensor[:max_len]
    
def prepare_for_training(model,
                         base_learning_rate: float,
                         new_learning_rate: float,
                         weight_decay: float):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        if "action_transformer" or "visual_transformer" or "MAF_layer" or "DMP_layer" or "fusion_layer" in name:
            new_params_list.append(param)
        else:
            base_params_list.append(param)

    optimizer = AdamW(
        [
            {'params': base_params_list,'lr': base_learning_rate, 'weight_decay': weight_decay},
            {'params': new_params_list,'lr': new_learning_rate, 'weight_decay': weight_decay}
        ],
        lr=base_learning_rate,
        weight_decay=weight_decay
    )

    del base_params_list
    del new_params_list
    gc.collect()
    torch.cuda.empty_cache()

    return optimizer

def check_and_create_directory(path_to_folder):
    """
    check if a nested path exists and create
    missing nodes/directories along the route
    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    return path_to_folder

def load_img_db(path_to_file: str) -> Optional[pd.DataFrame]:
    img_db = load_from_pickle(path_to_file)
    # re-format the keys
    img_db["path_to_image_file"] = img_db.apply(
        lambda row: "/".join(row["path_to_image_file"].split("/")[-2:]), 
        axis=1
    )
    return img_db