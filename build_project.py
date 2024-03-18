import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Project Setup')
    
    parser.add_argument('--project_dir', type=str, required=True, help='Path to GitHub repo')
    
    return parser.parse_args()
    


def build_project(args):
    
    print(args)
    



if __name__ == "__main__":
    
    args = parse_args()
    
    build_project(args)