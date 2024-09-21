from scraping.scraping_util import ScrapingUtil
import logging
import time

'''
Module to handle all functionality regarding specifically scraping the 
pro-football-reference (https://pro-football-reference.com) pages for 
relevant team and player metrics 
'''
class ProFootballReferenceScraper: 
   def __init__(self, teams, urls, year): 
      self._teams = teams 
      self._urls = urls
      self._year = year
     
       
   # Iniate scraping and construction of raw datasets regarding player and team metrics 
   def scrape(self): 
      home_page = self._urls["home-page"]
      scraping_util = ScrapingUtil() 
      logging.info(f"Beginning inital scraping process for the website Pro Football Reference ({home_page})")
      team_metrics_html = self.fetch_team_metrics(scraping_util)   
      logging.info(f"Team Metrics Size: {len(team_metrics_html)}")
      
      
   # Functionality to fetch raw HTML from each NFL Team Page 
   def fetch_team_metrics(self, scraping_util): 
      team_metrics_html = []
      for team in self._teams: 
         team_url = self.construct_team_url(self._urls['team-metrics'], team['acronym'])  
         logging.info(f"Fetching metrics for NFL Team {team['name']} via the URL {team_url}")     
         raw_html = scraping_util.fetchPage(team_url)   
         team_metrics_html.append(raw_html)  
      return team_metrics_html     
         
     
     
   # Functionality to construct proper NFL Team URL to fetch metrics from
   def construct_team_url(self, url, team_acronym): 
      return url.replace("{TEAM_ACRONYM}", team_acronym).replace("{CURRENT_YEAR}", self._year)
         
   

            
         
         