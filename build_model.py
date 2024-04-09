import argparse
import os
from model import MambaAudioModel
import yaml
import re
import torch

def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Audio Model Setup')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    parser.add_argument('--configs', type=str, required=False, default='base_configs.yaml', help='Filename of configuration directory')
    
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

    for key, value in configs.items():
        setattr(args, key, value)

    return args

def build_model(args):

    model = MambaAudioModel(args).to(args.device)
    optimizier = torch.optim.AdamW(model.parameters(),lr=args.lr)

if __name__ == "__main__":
    
    args = parse_args()
    
    args = load_configs(args)
    
    build_model(args)
