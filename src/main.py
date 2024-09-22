import logging
from config.yaml_config import YamlConfig
from config.logger_config import LoggerConfig
from scraping.pro_football_reference_scraper import ProFootballReferenceScraper

'''
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
'''
def main():
   #Set Up Global Logger Configurations
   logConfig = LoggerConfig(logging.INFO, "%(asctime)s - %(levelname)s - %(message)s")
   
   #Load Configurations
   config = YamlConfig("./config/application.yaml")

   
   #Initiate Scraping for Team/Player Metrics
   pfrScraper = ProFootballReferenceScraper(config.get_nfl_teams(), config.get_pro_football_reference_urls(), config.get_current_year())
   pfrScraper.scrape()
   


# Entry point for the script
if __name__ == "__main__":
    main()