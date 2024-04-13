import argparse
import os
import yaml

def parse_args():
    
    parser = argparse.ArgumentParser(description='Mamba-Music Project Setup')
    
    parser.add_argument('--project_name', type=str, required=True, help='Name of project')
    parser.add_argument('--project_path', type=str, required=True, help='Parent directory of project')
    
    return parser.parse_args()
    

def build_project(args):
    
    project_name = args.project_name
    project_path = args.project_path
    
    project_dir = os.path.join(project_path, project_name)
    
    sub_dirs = ['boards', 'inputs', 'generations','models', 'configs', 'encodings','tokenized']
    
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    else:
        print("Directory already exists.")
    
    for directory in sub_dirs:
        directory_path = os.path.join(project_dir, directory)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
    yaml_path = os.path.join(project_dir,'configs/configs.yaml')
    
    if not os.path.exists(yaml_path):
        with open(yaml_path, 'w') as file:  
            yaml.dump({}, file)



if __name__ == "__main__":
    
    args = parse_args()
    
    build_project(args)