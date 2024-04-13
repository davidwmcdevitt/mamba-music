import argparse
import os
from model import MambaAudioModel
import yaml
import re
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Audio Model Setup')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    parser.add_argument('--model_dict', type=str, required=True, help='Filename of model state dictionary')
    parser.add_argument('--configs', type=str, required=True, default='base_configs.yaml', help='Filename of configuration dictionary')
    
    parser.add_argument('--prompt', type=str, required=True, default='sample', choices=['noise', 'sample'], help='Context for the model')
    parser.add_argument('--prompt_length', type=int, required=True, default=1, help='Prompt duration in seconds')
    
    return parser.parse_args()

def load_configs(args):
    
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)
    configs_dir = os.path.join(project_dir,'configs')
        
    with open(os.path.join(configs_dir,args.configs), 'r') as config_file:
        configs = yaml.safe_load(config_file)

    for vocab_file in os.listdir(configs_dir):
        if vocab_file.startswith('vocab_'):
            configs['vocab_file'] = vocab_file
            configs['vocab_size'] = int(re.search(r'\d+', vocab_file).group())
            
    configs['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    configs['lr'] = float(configs['lr'])
    configs['datetime'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    for key, value in configs.items():
        setattr(args, key, value)
        
    print(args)
    return args

def load_model(args):
    
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)

    model = MambaAudioModel(args).to(args.device)

    checkpoint_path = os.path.join(project_dir,'models', args.model_dict)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
        
    print(f"Model {args.model_dict} loaded successfully")
    return model  

if __name__ == "__main__":
    
    args = parse_args()
    
    args = load_configs(args)
    
    model = load_model(args)
    
    
