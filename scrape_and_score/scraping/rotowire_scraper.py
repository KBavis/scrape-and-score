import pandas as pd
import requests 
import logging
from config import props
from service import team_service
from db import insert_data

'''
Fetch all historical odds for the current year

Args:
   None 

Returns:
   None
'''
def scrape_all(): 
   # fetch configs
   url = props.get_config('website.rotowire.urls.historical-odds') 
   curr_year = props.get_config('nfl.current-year')
   
   # retrieve historical odds
   jsonData = requests.get(url).json()
   df = pd.DataFrame(jsonData)
   
   # generate team betting odd records to persist for current year
   curr_year_data = df[df['season'] == str(curr_year)] 
   team_betting_odds_records = get_team_betting_odds_records(curr_year_data) 
   
   # insert into our db
   logging.info('Inserting all teams historical odds into our database')
   insert_data.insert_teams_historical_odds(team_betting_odds_records)
   
   
'''
Generate team betting odds records to persist into our database

Args:
   curr_year_data (pd.DataFrame): betting odds corresponding to current year
''' 
def get_team_betting_odds_records(curr_year_data: pd.DataFrame):
   # create team id mapping 
   mapping = create_team_id_mapping(curr_year_data[curr_year_data['week'] == '1'])
   betting_odds_records = [
   {
      'home_team_id': mapping[row['home_team_stats_id']],
      'away_team_id': mapping[row['visit_team_stats_id']],
      'home_team_score': row['home_team_score'],
      'away_team_score': row['visit_team_score'],
      'week': row['week'],
      'year': row['season'],
      'game_over_under': row['game_over_under'],
      'favorite_team_id': mapping[row['favorite']],
      'spread': row['spread'],
      'total_points': row['total'],
      'over_hit': row['over_hit'],
      'under_hit': row['under_hit'],
      'favorite_covered': row['favorite_covered'],
      'underdog_covered': row['underdog_covered'],
   }
   for _, row in curr_year_data.iterrows()
]
   return betting_odds_records
   

'''
Create a mapping of a teams acronym to corresponding team ID in our database

Args:
   week_one_data (pd.DataFrame): data corresponding to week one of current NFL season 

Returns:
   mapping (dict): mapping of a teams acronmy to team_id in our db
'''
def create_team_id_mapping(week_one_data: pd.DataFrame):
   # load configs 
   teams = props.get_config('nfl.teams')
   mappings = {}
   
   for team in teams:
      acronym = team.get('rotowire', team['pfr_acronym'])
      mappings[acronym] = team_service.get_team_id_by_name(team['name'])
      
   return mappings

'''
Fetch odds for the most recent week 

Args:
   None

Returns:
   None
'''
def fetch_recent(): 
   # determine recent week 
   logging.info('implement me')