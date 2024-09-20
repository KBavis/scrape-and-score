'''
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
'''

import logging 
from scraping.scraping_utility import ScrapingUtility

def main():
   #TODO: Create Logging Config File 
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
   logging.info("Beginning scraping process") 
   
   #Scrape for each URL 
   scrapingUtility = ScrapingUtility()
   htmlText = scrapingUtility.fetchPage("https://www.pro-football-reference.com/") #TODO: Add URLs to Application YAML File 
   


# Entry point for the script
if __name__ == "__main__":
    main()