import logging
import time
from pro_football_reference_web_scraper import player_game_log as p 
from pro_football_reference_web_scraper import team_game_log as t 
from nfl_data_py import import_depth_charts as roster


'''
Module to handle all functionality regarding specifically scraping the 
pro-football-reference (https://pro-football-reference.com) pages for 
relevant team and player metrics. We will utilize the following library 
(https://github.com/mjk2244/pro-football-reference-web-scraper) to not 
duplicate the effort needed to scrape the site. 
'''
class ProFootballReferenceScraper: 
   def __init__(self, teams, urls, year): 
      self._teams = teams 
      self._urls = urls
      self._year = year
     
       
   # Iniate scraping and construction of raw datasets regarding player and team metrics 
   def scrape(self): 
      # Fetch metrics for each of the 32 NFL Teams 
      
      # Fetch metrics for each QB/TE/WR/RB on the active 53 Man Roster for each NFL Team 
      
      # Generate raw CSV of team data 
      
      # Generate raw CSV of player data 
      
      #EX: TODO: Update me
      josh_allen_player_game_log = p.get_player_game_log(player = 'Josh Allen', position = 'QB', season = 2024)
      print(josh_allen_player_game_log)
      
      
      
      r
      
   
   '''
   Functionality to fetch relevant metrics corresponding to a specific NFL team
   
   Args: 
      None 
      
   Returns:
      pandas.DataFrame: A pandas DataFame with relevant metrics correspondingto the specific player     
   ''' 
   def fetch_team_metrics():
      logging.info("Fetching metrics for each NFL team")   

   '''
   Functionality to fetch the relevant metrics to a specific player
   
   Args:
      player (str) : NFL players full name
   
   Returns:
      pandas.DataFrame: A pandas DataFrame with relevant metrics corresponding to the specific player
   
   '''
   def fetch_player_metrics(name: str):
      

         
   

            
         
         