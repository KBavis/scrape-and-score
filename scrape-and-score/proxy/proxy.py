import requests 
import os
import logging 
import json
from datetime import datetime, timedelta
from random import choice

'''
Functionality to fetch relevant proxies 

Args: 
   None 
   
Returns: 
   proxies (json) - json object containing relevant proxies    
'''
def fetch_proxies(url, max=10):
   logging.info(f"Fetching proxies from the URL \'{url}\'")
   raw_json = requests.get(url).json()
   proxies = []
   count = 0
   
   for ip in raw_json['proxies']:
      if(count > max): 
         break 
      
      proxy = {"http": ip["proxy"]}
      if validate_proxy(proxy):
         count+=1
         proxies.append(proxy)
      else: 
         logging.info(f"Invalid Proxy: {proxy}")   
         continue 
   
   cache_proxies(proxies)   
   return proxies   


'''
Functionality to cache the fetched proxies via a 
'proxies.json' file  

Args: 
   proxies (list(dict)) 

Returns: 
   None   
'''
def cache_proxies(proxies):
   # configure cache data
   expiration = datetime.now() + timedelta(hours=3) 
   cache_data = {
      "expiration": expiration.isoformat(),
      "proxies": proxies
   }
   
   # create cache or override data
   with open("./resources/proxies.json", "w") as f:
      json.dump(cache_data, f, indent=4)

'''
Helper function to determine if proxies in cache have expired 

Args: 
   None
   
Returns:
   is_expired (bool) - truthy value indicating if cache is expired   
'''
def is_cache_expired():
   try: 
      with open("./resources/proxies.json") as f:
         cache_data = json.load(f) 
      
      expiration_time = datetime.fromisoformat(cache_data['expiration'])
      return datetime.now() >= expiration_time   
   except Exception as e: 
      logging.error(f"An error occured while checking the caches expiration: {e}")      
              

'''
Functionality to determine whether or not our proxy is valid 

Args:
   proxy (str) - proxy to validate 
   
Returns:
   valid (bool) - validity of proxy   
'''
def validate_proxy(proxy):
   try:
      ip = requests.get("http://checkip.amazonaws.com", proxies = proxy, timeout=2) 
      ip.raise_for_status()
      
      # valid IPv4 address contains three '.'
      if ip.text.count(".") == 3:
         logging.info(f"The proxy {proxy} is valid")
         return True 
      return False 
   except Exception as e:
      logging.info(f"An exception occured while checking if the proxy \'{proxy}\' is valid: {e}")
      return False   


'''
Utility function to fetch a proxy from our cache at random 

Args:
   None
   
Returns:
   proxy(dict) - random proxy in format {<protocol>:<proxy}   
'''
def get_random_proxy():
   with open("./resources/proxies.json", "r") as f:
      cache_data = json.load(f)
      
   return choice(cache_data['proxies'])   

'''
Main function that will initate the logic regarding fetching and setting up of proxies 

Args:
   url (str) - URL pertaining to where we should be fetching proxies from 
   
Returns:
   proxy (dict) - random proxy in format {<protocol>: <proxy>}

'''
def get_proxy(url = "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=http&proxy_format=ipport&format=json"): 
   # fetch proxies if none are cached
   if not os.path.exists("./resources/proxies.json"):
      logging.info("No cached proxies exist; attempting to fetch proxies")
      fetch_proxies(url)
   
   # fetch proxies if cache expired
   if is_cache_expired():
      logging.info("Cache is expired; attempting to update proxies")
      fetch_proxies(url)    
      
   return get_random_proxy()        