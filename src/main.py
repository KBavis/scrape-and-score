import logging
from config.yaml_config import YamlConfig
from scraping.scraping_utility import ScrapingUtility
from config.logger_config import LoggerConfig

'''
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
'''
def main():
   #Load Configurations
   config = YamlConfig("./config/application.yaml")
    
   #Create Logger 
   logConfig = LoggerConfig(logging.INFO, "%(asctime)s - %(levelname)s - %(message)s")
   log = logConfig.get_logger()
   
   #Scrape for each URL 
   scrapingUtility = ScrapingUtility()
   for url in config.get_websites(): 
      log.info(f"Fetching HTML content for the following URL: {url}")
      htmlText = scrapingUtility.fetchPage(url);
   


# Entry point for the script
if __name__ == "__main__":
    main()