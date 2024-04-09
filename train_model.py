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
    
    parser.add_argument('--continue_train', action='store_false', required=False, help='Continue training')
    parser.add_argument('--checkpoint_name', type=str, required=False, default='base_configs.yaml', help='Filename of checkpoint state dict')
    
    
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

    for key, value in configs.items():
        setattr(args, key, value)

    print(args)
    return args

def build_model(args):
    
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)

    model = MambaAudioModel(args).to(args.device)
    print(model)
    
    if args.continue_train:
        checkpoint_path = os.path.join(project_dir,'checkpoint', args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        
    return model  

if __name__ == "__main__":
    
    args = parse_args()
    
    args = load_configs(args)
    
    model = build_model(args)
    
    optimizier = torch.optim.AdamW(model.parameters(),lr=args.lr)
