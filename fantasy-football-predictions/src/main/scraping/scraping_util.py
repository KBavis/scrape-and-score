import requests
import logging 
from swiftshadow import QuickProxy

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
      proxy = QuickProxy()
      response = requests.get(url, proxies={proxy[1]:proxy[0]})
      response.raise_for_status()
      return response.text
   except requests.RequestException as e:
      print(f"Error while fetching HTML content from the following URL {url} : {e}")
      return None 
   
   
   