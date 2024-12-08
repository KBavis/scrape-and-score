'''
Module to act as intermediary between business logic & data access layer for player_game_logs 
'''
import logging
from db import insert_data, fetch_data
import pandas as pd
from . import team_service, service_util

'''
Functionality to insert multiple players game logs 

Args:
   player_metrics (list): list of dictionaries containing player's name, position, and game_logs
   depth_charts (list): list of dictionaries containing a plyer's name & corresponding player_id

Returns:
   None
'''
def insert_multiple_players_game_logs(player_metrics: list, depth_charts: list):
   for player in player_metrics:
      player_name = player['player']
      logging.info(f"Attempting to insert game logs for player '{player_name}'")
      
      # extract position 
      position = player['position']
      
      # ensure metrics recorded
      df = player['player_metrics']
      if df.empty: 
         logging.warning(f"No player metrics corresponding to player '{player_name}; skipping insertion")
         continue
      
      # fetch players ID 
      player_id = get_player_id_by_name(player_name, depth_charts)
      
      # iterate through game logs, generate tuples, and insert into db
      tuples = []
      if position == 'QB':
         tuples = get_qb_game_log_tuples(df, player_id)
         insert_data.insert_qb_player_game_logs(tuples)
      elif position == 'RB':
         tuples = get_rb_game_log_tuples(df, player_id)
         insert_data.insert_rb_player_game_logs(tuples)
      elif position == 'TE' or position == 'WR':
         tuples = get_wr_or_te_game_log_tuples(df, player_id)
         insert_data.insert_wr_or_te_player_game_logs(tuples)
      else:
         raise Exception(f"Unknown position '{position}'; unable to fetch game log tuples")
      
      

def is_player_game_logs_empty(): 
   player_game_log = fetch_data.fetch_one_player_game_log() 
   
   if player_game_log == None:
      return True
   else:
      return False


# TODO (FFM-84): Refactor this code so that utility functions in seperate file

'''
Utility function to retrieve a players ID based on their name 

Args:
   depth_chart (list): list of dictionary items to iterate through and retrieve id for 
   player_name (str): the player name to retrieve ID for 
   
Returns:
   player_id (int): id corresponding to the specified player
'''
def get_player_id_by_name(player_name: str, depth_charts: list): 
   player = next((player for player in depth_charts if player["player_name"] == player_name), None)
   return player["player_id"] if player else None


'''
Utility function to fetch game log tuples to insert into our database for a QB

Args: 
   df (pd.DataFrame): data frame to extract into tuples

Returns:
   tuples (list): list of tuples to be directly inserted into our database
'''
def get_qb_game_log_tuples(df: pd.DataFrame, player_id: int):
   tuples = []
   for _, row in df.iterrows():
      game_log = (
         player_id, 
         row['week'], \
         service_util.extract_day_from_date(row['date']), 
         service_util.extract_year_from_date(row['date']), 
         service_util.extract_home_team_from_game_location(row['game_location']), 
         team_service.get_team_id_by_name(service_util.get_team_name_by_pfr_acronym(row['opp'])), 
         row['result'], 
         row['team_pts'], 
         row['opp_pts'], 
         row['cmp'], 
         row['att'], 
         row['pass_yds'], 
         row['pass_td'], 
         row['int'], 
         row['rating'], 
         row['sacked'], 
         row['rush_att'], 
         row['rush_yds'], 
         row['rush_td']
      )
      tuples.append(game_log)
   
   return tuples

'''
Utility function to fetch game log tuples to insert into our database for a RB

Args: 
   df (pd.DataFrame): data frame to extract into tuples

Returns:
   tuples (list): list of tuples to be directly inserted into our database
'''
def get_rb_game_log_tuples(df: pd.DataFrame, player_id: int):
   tuples = []
   for _, row in df.iterrows():
      game_log = (
         player_id, 
         row['week'], 
         service_util.extract_day_from_date(row['date']),
         service_util.extract_year_from_date(row['date']),
         service_util.extract_home_team_from_game_location(row['game_location']),
         team_service.get_team_id_by_name(service_util.get_team_name_by_pfr_acronym(row['opp'])), 
         row['result'],
         row['team_pts'],
         row['opp_pts'],
         row['rush_att'], 
         row['rush_yds'], 
         row['rush_td'], 
         row['tgt'], 
         row['rec'], 
         row['rec_yds'], 
         row['rec_td']
      )
      tuples.append(game_log)
   
   return tuples

'''
Utility function to fetch game log tuples to insert into our database for a WR or TE

Args: 
   df (pd.DataFrame): data frame to extract into tuples

Returns:
   tuples (list): list of tuples to be directly inserted into our database
'''
def get_wr_or_te_game_log_tuples(df: pd.DataFrame, player_id: int):
   tuples = []
   for _, row in df.iterrows():
      game_log = (
         player_id, 
         row['week'], 
         service_util.extract_day_from_date(row['date']), 
         service_util.extract_year_from_date(row['date']), 
         service_util.extract_home_team_from_game_location(row['game_location']), 
         team_service.get_team_id_by_name(service_util.get_team_name_by_pfr_acronym(row['opp'])), 
         row['result'], 
         row['team_pts'],
         row['opp_pts'], 
         row['tgt'], 
         row['rec'], 
         row['rec_yds'], 
         row['rec_td'], 
         row['snap_pct']
      )
      tuples.append(game_log)
   
   return tuples



   
   