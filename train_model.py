import argparse
import os
from model import MambaAudioModel
import yaml
import re
import torch
import random
import numpy as np
from tqdm import tqdm
import psutil
from datetime import datetime

def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Audio Model Setup')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    parser.add_argument('--configs', type=str, required=False, default='base_configs.yaml', help='Filename of configuration directory')
    
    parser.add_argument('--continue_train', action='store_true', required=False, default=False, help='Continue training')
    parser.add_argument('--checkpoint_name', type=str, required=False, help='Filename of checkpoint state dict')
    
    parser.add_argument('--model_name', type=str, required=False, help='Filename of model state dict')
    parser.add_argument('--save_frequency', type=int, required=False, help='Save model every X epochs')
    
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

def build_model(args):
    
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)

    model = MambaAudioModel(args).to(args.device)
    print(model)
    
    if args.continue_train:
        checkpoint_path = os.path.join(project_dir,'models', args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        
    return model  

def prepare_tracks(args):
    
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)
    tracks_dir = os.path.join(project_dir,'tokenized')
    
    if len(os.listdir(tracks_dir)) > args.load_cap:
        
        track_list = random.sample(os.listdir(tracks_dir), args.load_cap)
        
    else:
        
        track_list =os.listdir(tracks_dir)
        
    for track_name in track_list:
            
        if track_name == track_list[0]:
            tokenized_tracks = [torch.load(os.path.join(tracks_dir,track_name))]
        else:
            tokenized_track = torch.load(os.path.join(tracks_dir,track_name))
            tokenized_tracks.append(tokenized_track)
                
    return tokenized_tracks

def save_model(model, args):
    
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)
    model_dir = os.path.join(project_dir,'models')
    
    if args.model_name:
        model_filename = args.model_name + '.pt'
    elif args.continue_train:
        model_filename = args.checkpoint_name + '.pt'
    else:
        model_filename = f"{args.project_name}_{args.datetime}.pt"
    
    model_path = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    
    print("Model saved as:", model_path)
    

if __name__ == "__main__":
    
    args = parse_args()
    
    args = load_configs(args)
    
    model = build_model(args)
    
    optimizier = torch.optim.AdamW(model.parameters(),lr=args.lr)
    
    loss_list = []
    
    system_memory = []
    
    gen_audio_array = np.zeros(((48000, 2)))
    gen_audio_spec = np.zeros((128, 94))
    
    generated_tokens = torch.zeros((1, 1), dtype=torch.int64)

    for epoch in range(args.num_epochs):
    
        tracks = prepare_tracks(args)

        print(f"Epoch {epoch}")
        model.train()

        system_memory.append(psutil.virtual_memory().percent)

        for _ in tqdm(range(args.iterations)):
            
            batch = []

            for _ in range(args.batch_size):

                track = tracks[random.randint(0,len(tracks)-1)]

                starting_point = random.randint(0,len(track)-args.block_size)

                clip = track[starting_point:starting_point+args.block_size]

                batch.append(torch.tensor(clip, dtype=torch.long))

            batch = torch.stack(batch)

            batch_input = batch[:,:-1].contiguous().to(args.device)
            batch_target = batch[:,1:].contiguous().to(args.device)

            logits, loss = model(batch_input, batch_target)

            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)

            optimizier.step()

            optimizier.zero_grad()

            loss_list.append(loss.item())

            del batch_input, batch_target, logits, loss
            torch.cuda.empty_cache()
            
        if args.save_frequency:
            
            if epoch % 5 == 0:
                
                save_model(model, args)
                
    save_model(model, args)
            
    

