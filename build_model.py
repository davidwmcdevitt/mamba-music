import argparse
import os
from model import MambaAudioModel
import yaml

def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Audio Model Setup')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    parser.add_argument('--configs', type=str, required=True, help='Filename of configuration directory')
    
    return parser.parse_args()

def load_configs(args):
        
    with open(os.path.join(args.project_path,'configs',args.configs), 'r') as config_file:
        configs = yaml.safe_load(config_file)
    

def build_model():
    pass

if __name__ == "__main__":
    
    args = parse_args()
    
    load_configs(args)
    
    build_model(args)
