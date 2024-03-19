import argparse
import os
from pytube import Playlist, YouTube
from moviepy.editor import AudioFileClip
from pytube.exceptions import AgeRestrictedError
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Audio Retrieval')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    parser.add_argument('--source', type=str, required=True, default='youtube', help='Audio source')
    parser.add_argument('--url', type=str, required=True, help='Youtube URL (playlist or video)')
    
    return parser.parse_args()

def is_playlist(url):
    return bool(re.search(r'\bplaylist\b', url, re.IGNORECASE))

def fetch_audio(args):
        
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)
        
    source = args.source
    
    if source == 'youtube':
        
        url = args.url
        
        if is_playlist(url):
            
            playlist = Playlist(url)
            
            for video in tqdm(playlist.videos):
                
                output_path = os.path.join(project_dir, 'inputs')
                
                mp4_name = video.title.replace('/', '') + ".mp4"
                wav_name = video.title.replace('/', '') + ".wav"
                
                if wav_name not in os.listdir(output_path) and mp4_name not in os.listdir(output_path):
                
                    try:
                        audio_stream = video.streams.get_audio_only()
                        audio_stream.download(output_path=output_path, filename=mp4_name)
                    except AgeRestrictedError:
                        print(f"Video {video.title} is age restricted and cannot be downloaded.")
                                            
                                        
                    clip = AudioFileClip(os.path.join(output_path,mp4_name))
                    clip.write_audiofile(os.path.join(output_path,wav_name))
                    os.remove(os.path.join(output_path,mp4_name))
                        
        else:
            
            video = YouTube(url)
                
            output_path = os.path.join(project_dir, 'inputs')
            
            mp4_name = video.title.replace('/', '') + ".mp4"
            wav_name = video.title.replace('/', '') + ".wav"
            
            if wav_name not in os.listdir(output_path) and mp4_name not in os.listdir(output_path):
            
                try:
                    audio_stream = video.streams.get_audio_only()
                    audio_stream.download(output_path=output_path, filename=video.title.replace('/', '') + ".mp4")
                except AgeRestrictedError:
                    print(f"Video {video.title} is age restricted and cannot be downloaded.")
                                        
                                    
                clip = AudioFileClip(os.path.join(output_path,mp4_name))
                clip.write_audiofile(os.path.join(output_path,wav_name))
                os.remove(os.path.join(output_path,mp4_name))
                
    
            

if __name__ == "__main__":
    
    args = parse_args()
    
    fetch_audio(args)