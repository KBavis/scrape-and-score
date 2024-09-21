import requests
from bs4 import BeautifulSoup
import logging 

'''
Class to contain all the generic scraping functionality that we will need to utilize when 
scraping each individual website 
'''
class ScrapingUtil: 
   
   '''Functionality to fetch the HTMLL Content from the specified URL'''
   def fetchPage(self, url):   
      try:
         logging.info(f"Fetching HTML Content from the URL {url}")
         response = requests.get(url)
         response.raise_for_status()
         return response.text
      except requests.RequestException as e:
         print(f"Error fetching HTML Content from URL {url}: {e}")
         return None