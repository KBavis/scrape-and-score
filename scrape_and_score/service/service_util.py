from config import props
import logging


'''
Utility function to extract the day from a string date 

Args:
   date (str): date to extract day from 

Returns:
   day (str): day corresponding to date 
'''
def extract_day_from_date(date: str): 
   split = date.split('-')
   return split[2]


'''
Utility function to extract the year from a string date 

Args:
   date (str): date to extract year from 

Returns:
   year (str): year corresponding to date 
'''
def extract_year_from_date(date: str):
   split = date.split('-')
   return split[0]


'''
Utility function to extract whether or not player was at home 

Args:
   game_location (str): either an @ sign or nothing 

Returns: 
   bool (str): T or F indicating home_team or not 
'''
def extract_home_team_from_game_location(game_location: str): 
   if game_location == '@':
      return 'F'
   else:
      return 'T'
   


'''
Utility function to extract team name from PFR Acronym from properties file 

Args:
   opp (str): acronym to fetch team for 

Returns:
   team_name (str): team name corresponding to acronym
'''
def get_team_name_by_pfr_acronym(opp: str):
   teams = props.get_config('nfl.teams', [])
   
   for team in teams:
      if team.get('pfr_acronym') == opp:
         return team.get('name')
      
   logging.warning(f"No team found for acronym: '{opp}'")
   return None

'''
Utility function to determine which year a game log should have

Args:
   week (str): numeric corresponding to week
   year (int): current year we are fetching metrics for
'''
def get_game_log_year(week: str, year: int):
   if week == '18':
      return year + 1 # next year only if past week 18
   else :
      return year