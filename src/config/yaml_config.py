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
   
   # Fetch relevant URLs for Pro Football Reference 
   def get_pro_football_reference_urls(self):
      return self._config['website']['pro-football-reference']['urls']
   
   # Fetch all NFL Teams 
   def get_nfl_teams(self):
      return self._config['nfl']['teams']
   
   # Fetch current year of NFL Season
   def get_current_year(self):
      return self._config['nfl']['current-year']
   