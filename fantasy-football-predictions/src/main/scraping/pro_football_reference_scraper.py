import logging
import time
from pro_football_reference_web_scraper import player_game_log as p 
from pro_football_reference_web_scraper import team_game_log as t 

'''
Module to handle all functionality regarding specifically scraping the 
pro-football-reference (https://pro-football-reference.com) pages for 
relevant team and player metrics. We will utilize the following library 
(https://github.com/mjk2244/pro-football-reference-web-scraper) to not 
duplicate the effort needed to scrape the site. 
'''       

# Fetch metrics for all players and NFL teams in current seasons
def fetch_metrics(team_data): 
    for team in team_data: 
       print(team)
        # josh_allen_player_game_log = p.get_player_game_log(player='Josh Allen', position='QB', season=2024)
        # print(josh_allen_player_game_log)
        

'''
Functionality to fetch relevant metrics corresponding to a specific NFL team
   
Args: 
    team (str) - NFL team full name
      
Returns:
    pandas.DataFrame: A pandas DataFrame with relevant metrics corresponding to the specific player     
''' 
def fetch_team_metrics(team):
    logging.info("Fetching metrics for each NFL team")   
    return None

'''
Functionality to fetch the relevant metrics to a specific player
   
Args:
    player (str): NFL player's full name
   
Returns:
    pandas.DataFrame: A pandas DataFrame with relevant metrics corresponding to the specific player
'''
def fetch_player_metrics(name: str):
    return None


         
   

            
         
         