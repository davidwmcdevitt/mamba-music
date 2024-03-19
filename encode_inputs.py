import argparse
import os
from transformers import EncodecModel, AutoProcessor
import librosa
import warnings
import torch
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Audio Vocabulary Analysis')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    return parser.parse_args()

def encode_inputs(args):
        
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)

    encoder = EncodecModel.from_pretrained("facebook/encodec_48khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
    
    input_files = os.listdir(os.path.join(project_dir,'inputs'))
    
    for file_ in tqdm(input_files):
    
        tensor_name = os.path.splitext(file_)[0] + '.pt'
        
        if tensor_name not in os.listdir(os.path.join(project_dir,'encodings')):
        
            input_path = os.path.join(project_dir,'inputs', file_)
            
            audio_array, sample_rate = librosa.load(input_path, sr = processor.sampling_rate, mono = False)
            
            inputs = processor(raw_audio=audio_array, sampling_rate=processor.sampling_rate, return_tensors="pt")
            audio_codes = encoder.encode(inputs["input_values"], inputs["padding_mask"]).audio_codes
            
            torch.save(audio_codes, os.path.join(project_dir,'encodings', tensor_name + '.pt'))

    
def analyze_vocab(args):
    pass
    
    

if __name__ == "__main__":
    
    args = parse_args()
    
    encode_inputs(args)