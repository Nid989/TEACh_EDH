import json
import pickle
import yaml
from PIL import Image
import subprocess
import torch

def load_from_yaml(path_to_file: str):
    print(f"loading data from .yaml file @ {path_to_file}")
    with open(path_to_file) as file:
        _dict = yaml.safe_load(file)
    return _dict

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