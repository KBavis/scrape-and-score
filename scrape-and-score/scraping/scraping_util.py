import requests
import logging 
from swiftshadow.classes import Proxy
import time 
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
      #TODO: Implement Rotating Proxy Logic (FFM-12)
      logging.info(f"Fetching raw HTML from the following URL: {url}")
      time.sleep(3)
      response = session.get(url, proxies=get_proxy(), headers={"User-Agent":"PostmanRuntime/7.42.0"})
      response.raise_for_status()
      return response.text
   except requests.RequestException as e:
      print(f"Error while fetching HTML content from the following URL {url} : {e}")
      return None 

'''
Functionality to generate a proxy for each of our requests 

Args:
   None 
   
Returns: 
   proxies (dict) - dictionary containing proxies    
''' 
def get_proxy():
   swiftshadow = Proxy(autoRotate=True)
   proxy = swiftshadow.proxy() 
   proxies = {proxy[1]:proxy[0]}
   logging.info(f"Proxies generated: {proxies}")
   
      
      
   
   
   