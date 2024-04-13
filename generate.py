import argparse
import os
from model import MambaAudioModel
from transformers import EncodecModel, AutoProcessor
import yaml
import re
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from encode_inputs import tuple_lookup
import soundfile as sf

def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Audio Model Setup')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    parser.add_argument('--model_dict', type=str, required=True, help='Filename of model state dictionary')
    parser.add_argument('--configs', type=str, required=True, default='base_configs.yaml', help='Filename of configuration dictionary')
    
    parser.add_argument('--prompt', type=str, required=True, default='noise', choices=['noise', 'sample','input'], help='Context for the model')
    parser.add_argument('--prompt_length', type=int, required=True, default=1, help='Prompt duration in seconds')
    parser.add_argument('--keep_prompt', action='store_true', required=False, default=False, help='Keep prompt in generated audio file')
    parser.add_argument('--input_path', type=str, required=False, help='Prompt input file')
    
    parser.add_argument('--gen_length', type=int, required=True, default=1, help='Prompt duration in seconds')
    
    
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
            configs['vocab_path'] = os.path.join(configs_dir, vocab_file)
            
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

def load_sample(args):
    
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)
    tracks_dir = os.path.join(project_dir,'tokenized')
    
    track_list = os.listdir(tracks_dir)
    
    track = torch.load(os.path.join(tracks_dir,random.sample(track_list,1)[0]))

    starting_point = random.randint(0,len(track)-args.prompt_length)

    prompt = torch.tensor(track[starting_point:starting_point+args.prompt_length], dtype=torch.long).unsqueeze(0).to(args.device)
    
    return prompt
    
def load_input(args):
    pass    
    
def generate_noise(args):
    pass

if __name__ == "__main__":
    
    args = parse_args()
    
    args = load_configs(args)
    
    encoder = EncodecModel.from_pretrained("facebook/encodec_48khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
    
    model = load_model(args)
    
    model.eval()
    
    if args.prompt == 'sample':
        
        prompt = load_sample(args)
        
    elif args.prompt == 'input':
        
        prompt = load_input(args)
        
    elif args.prompt == 'noise':
        
        prompt = generate_noise(args)
        
    generated_tokens = torch.tensor([], dtype = torch.long)

    generation_length = 150 * args.gen_length

    for _ in tqdm(range(generation_length)):

        logits, loss = model(prompt)

        last_timestep = logits[:,-1,:]
        probs = F.softmax(last_timestep, dim=1)
        next_index = torch.multinomial(probs, num_samples=1)

        prompt = torch.cat((prompt, next_index), dim=1)[:,-args.block_size:]

        next_index = next_index.detach().cpu()

        generated_tokens = torch.cat((generated_tokens, next_index), dim=1)

    generated_tokens = generated_tokens.detach().cpu().numpy()
    
    generated_encodings = []
    
    vocab = np.load(args.vocab_path)
    
    for token in tqdm(generated_tokens.squeeze(0)):
    
        generated_encodings.append(tuple_lookup(token, vocab))
        
    acoustic_clip = []

    semantic_clip = []

    generated_audio_codes = torch.tensor([], dtype=torch.long)

    for encoding in generated_encodings:

        acoustic_clip.append(encoding[0])
        semantic_clip.append(encoding[1])

    acoustic_clip = torch.tensor(acoustic_clip, dtype=torch.long)
    semantic_clip = torch.tensor(semantic_clip, dtype=torch.long)

    generated_audio_codes = torch.stack((acoustic_clip, semantic_clip), dim=0).view(2,150 * args.gen_length).unsqueeze(0).unsqueeze(0)

    decoder_padding = torch.ones(1, processor.sampling_rate*(args.gen_length))

    decoder_outputs = encoder.decode(generated_audio_codes, [None], decoder_padding)

    generated_audio = decoder_outputs.audio_values.squeeze(0)

    generated_audio = generated_audio.detach().cpu().numpy()
    generated_audio = np.swapaxes(generated_audio, 0, 1)

    generation_path = os.path.join(args.project_path, args.project_name, 'generations', f'{args.project_name}_{args.gen_length}s_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.wav')

    sf.write(generation_path, generated_audio, processor.sampling_rate)
    print(f'Generation saved at {generation_path}')
        
        