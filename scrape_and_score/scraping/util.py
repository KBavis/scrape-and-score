import requests
import logging 
import time 
from proxy import proxy
from config import get_config
session = requests.Session() 

'''
Functionality to fetch the raw HTML corresponding to a particular URL 

Args:
   url - the URL we want to extract the raw HTML from

Returns:
   raw HTML corresponding to specified URL   
'''

def fetch_page(url: str):
   try:
      logging.info(f"Fetching raw HTML from the following URL: {url}")
      
      time.sleep(get_config('scraping.delay'))
      response = session.get(url, proxies=proxy.get_proxy())
      response.raise_for_status()
      return response.text
   except requests.RequestException as e:
      print(f"Error while fetching HTML content from the following URL {url} : {e}")
      return None 

   
      
      
   
   
   