import argparse
import os
from transformers import EncodecModel, AutoProcessor
import librosa
import warnings
import torch
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore")

def token_lookup(encoding_tuple, unique_tuples):
  return unique_tuples.index(encoding_tuple)

def tuple_lookup(token, unique_tuples):
  return unique_tuples[token]

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
    
    for file_ in tqdm(input_files, mininterval=10):
    
        tensor_name = os.path.splitext(file_)[0] + '.pt'
        
        if tensor_name not in os.listdir(os.path.join(project_dir,'encodings')):
        
            input_path = os.path.join(project_dir,'inputs', file_)
            
            audio_array, sample_rate = librosa.load(input_path, sr = processor.sampling_rate, mono = False)
            
            inputs = processor(raw_audio=audio_array, sampling_rate=processor.sampling_rate, return_tensors="pt")
            audio_codes = encoder.encode(inputs["input_values"], inputs["padding_mask"]).audio_codes
            
            torch.save(audio_codes, os.path.join(project_dir,'encodings', tensor_name))

    
def analyze_vocab_old(args):
        
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)
    encodings_dir = os.path.join(project_dir,'encodings')
    tokens_dir = os.path.join(project_dir,'tokenized')
    
    encodings = os.listdir(encodings_dir)
    
    for file_ in tqdm(encodings, mininterval=10):
        
        input_path = os.path.join(encodings_dir, file_)
          
        audio_codes = torch.load(input_path)
          
        if file_ == encodings[0]:
            sequence = audio_codes
        else:
            sequence = torch.cat((sequence, audio_codes), dim = 0)
          
        tupled_sequence = []
          
        for row in range(sequence.shape[0]):
            for digit in range(150):
                tupled_sequence.append((int(sequence[row,0,0,digit]), int(sequence[row,0,1,digit])))
          
        track_tuples = []
        for row in range(audio_codes.shape[0]):
            for digit in range(150):
                track_tuples.append(tuple((int(audio_codes[row,0,0,digit]), int(audio_codes[row,0,1,digit]))))
          
        if file_ == encodings[0]:
            tracks = [track_tuples]
        else:
            tracks.append(track_tuples)
        
    unique_tuples, counts = np.unique(tupled_sequence, axis=0, return_counts=True)
    
    unique_tuples = [tuple(row) for row in unique_tuples]
    
    vocab_size = len(unique_tuples)
    
    np.save(os.path.join(project_dir, f'configs/vocab_{vocab_size}.npy'), unique_tuples)
    
    for track, file_ in tqdm(zip(tracks, encodings)):
    
        if file_ not in os.listdir(tokens_dir):

            tokenized_track = [token_lookup(tuple_,unique_tuples) for tuple_ in track]
            
            torch.save(tokenized_track, os.path.join(project_dir,'tokenized', file_))


def analyze_vocab(args):
    project_name = args.project_name
    project_path = args.project_path

    project_dir = os.path.join(project_path, project_name)
    encodings_dir = os.path.join(project_dir, 'encodings')
    tokens_dir = os.path.join(project_dir, 'tokenized')
    if not os.path.exists(tokens_dir):
        os.makedirs(tokens_dir)

    encodings = os.listdir(encodings_dir)
    sequence_list = []
    tracks = []

    for file_ in tqdm(encodings, mininterval=10):
        input_path = os.path.join(encodings_dir, file_)
        audio_codes = torch.load(input_path)
        sequence_list.append(audio_codes)
        track_tuples = [(int(audio_codes[row, 0, 0, digit]), int(audio_codes[row, 0, 1, digit]))
                        for row in range(audio_codes.shape[0]) for digit in range(150)]
        tracks.append(track_tuples)

    sequence = torch.cat(sequence_list, dim=0)

    tupled_sequence = [(int(sequence[row, 0, 0, digit]), int(sequence[row, 0, 1, digit]))
                       for row in range(sequence.shape[0]) for digit in range(150)]

    unique_tuples, counts = np.unique(tupled_sequence, axis=0, return_counts=True)
    unique_tuples = [tuple(row) for row in unique_tuples]
    vocab_size = len(unique_tuples)
    np.save(os.path.join(project_dir, f'configs/vocab_{vocab_size}.npy'), unique_tuples)

    for track, file_ in tqdm(zip(tracks, encodings), mininterval=10):
        tokenized_path = os.path.join(tokens_dir, file_)
        if not os.path.exists(tokenized_path):
            tokenized_track = [token_lookup(tuple_, unique_tuples) for tuple_ in track]
            torch.save(tokenized_track, tokenized_path)
    
    
    
    

if __name__ == "__main__":
    
    args = parse_args()
    
    encode_inputs(args)
    
    analyze_vocab(args)