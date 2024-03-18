import argparse
import os
from pytube import Playlist, YouTube
from moviepy.editor import AudioFileClip
from pytube.exceptions import AgeRestrictedError
import re
import tqdm

def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Project Setup')
    
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
                output_path = os.path.join(project_dir, video.title.replace('/', '') + ".mp4")
                if os.path.exists(output_path):
                    continue
                try:
                    audio_stream = video.streams.get_audio_only()
                    audio_stream.download(output_path=output_path, filename=video.title.replace('/', '') + ".mp4")
                except AgeRestrictedError:
                    print(f"Video {video.title} is age restricted and cannot be downloaded.")
                    
        else:
            
            video = YouTube(url)
            output_path = os.path.join(project_dir, video.title.replace('/', '') + ".mp4")
            if os.path.exists(output_path):
                pass
            else:
                try:
                    audio_stream = video.streams.get_audio_only()
                    audio_stream.download(output_path=output_path, filename=video.title.replace('/', '') + ".mp4")
                except AgeRestrictedError:
                    print(f"Video {video.title} is age restricted and cannot be downloaded.")
                

            

if __name__ == "__main__":
    
    args = parse_args()
    
    fetch_audio(args)