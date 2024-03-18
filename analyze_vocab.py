import argparse
import os
from transformers import EncodecModel, AutoProcessor
import librosa
import yaml


def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Audio Vocabulary Analysis')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    return parser.parse_args()

def analyze_vocab(args):
        
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)
    
    if os.path.exists(os.path.join(project_path, 'configs','vocab.yaml')):
        
        encoder = EncodecModel.from_pretrained("facebook/encodec_48khz")
        processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
        
        sequence = []
        
        input_files = os.listdir(os.path.join(project_dir,'inputs'))
        
        for file_ in input_files:
            
            input_path = os.path.join(project_dir,'inputs', file_)
            
            audio_array, sample_rate = librosa.load(input_path, sr = processor.sampling_rate, mono = False)
            
            inputs = processor(raw_audio=audio_array, sampling_rate=processor.sampling_rate, return_tensors="pt")
            audio_codes = encoder.encode(inputs["input_values"], inputs["padding_mask"]).audio_codes
            
            print(audio_codes.shape)

        

if __name__ == "__main__":
    
    args = parse_args()
    
    analyze_vocab(args)