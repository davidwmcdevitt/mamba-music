import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Project Setup')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    return parser.parse_args()
    


def build_project(args):
    
    print(args)
    



if __name__ == "__main__":
    
    args = parse_args()
    
    build_project(args)