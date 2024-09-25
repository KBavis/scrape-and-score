import logging;

class LoggerConfig:
   def __init__(self, level, format): 
      self._level = level
      self._format = format 
      self.configure_logger() 
   
   def configure_logger(self):
      logging.basicConfig(level=self._level, format=self._format)
