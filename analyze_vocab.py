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
        
        for files in input_files:
            print(files)
        

if __name__ == "__main__":
    
    args = parse_args()
    
    analyze_vocab(args)