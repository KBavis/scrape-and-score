import requests
import logging 

'''
Functionality to fetch the raw HTML corresponding to a particular URL 

Args:
   url - the URL we want to extract the raw HTML from

Returns:
   raw HTML corresponding to specified URL   
'''
def fetch_page(url):
   try:
      logging.info(f"Fetching HTML Content from the following URL: {url}")
      response = requests.get(url)
      response.raise_for_status()
      return response.text
   except requests.RequestException as e:
      print(f"Error while fetching HTML content from the following URL {url} : {e}")
      return None 
   
   
   