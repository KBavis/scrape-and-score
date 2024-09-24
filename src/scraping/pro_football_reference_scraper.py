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
class ProFootballReferenceScraper: 
   def __init__(self, teams, urls, year): 
      self._teams = teams 
      self._urls = urls
      self._year = year
     
       
   # Iniate scraping and construction of raw datasets regarding player and team metrics 
   def scrape(self): 
      josh_allen_player_game_log = p.get_player_game_log(player = 'Josh Allen', position = 'QB', season = 2024)
      print(josh_allen_player_game_log)


         
   

            
         
         