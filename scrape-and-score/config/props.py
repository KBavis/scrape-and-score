import yaml

def load_configs(file_path="./resources/application.yaml"):
   with open(file_path, 'r') as file:
      return yaml.safe_load(file)