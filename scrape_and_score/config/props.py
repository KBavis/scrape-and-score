import yaml

'''
Functionality to load our configurations 

Args: 
   file_path(str): path containing configurations 

Returns:
   loaded configurations
'''
def load_configs(file_path="./resources/application.yaml"):
   with open(file_path, 'r') as file:
      return yaml.safe_load(file)