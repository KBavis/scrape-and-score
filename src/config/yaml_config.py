import yaml

'''
Class to load necessary configurations from our YAML File
'''
class YamlConfig:
   def __init__(self, file_path):
      self._config = self.load_configs(file_path)
     
   # Load all available YAML Configurations    
   def load_configs(self, file_path):
      with open(file_path, 'r') as file:
         return yaml.safe_load(file)    
   
   # Fetch all listed websites to scrape 
   def get_websites(self):
      return self._config['website']['urls']