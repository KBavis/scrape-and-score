import logging 
import pandas as pd
from . import service_util, team_service
from db import insert_data, fetch_data
from config import props

'''
Functionality to insert multiple teams game logs 

Args: 
   team_metrics (list): list of dictionaries containing team's name & game logs 
   teams_and_ids (list): list of dictionaries containing team's name & corresponding team ID 
   
Returns:
   None
'''
def insert_multiple_teams_game_logs(team_metrics: list, teams_and_ids: list): 
   
   curr_year = props.get_config('nfl.current-year') # fetch current year

   remove_previously_inserted_game_logs(team_metrics, curr_year, teams_and_ids)

   if len(team_metrics) == 0:
      logging.info('No new team game logs to persist; skipping insertion')
      return
   
   for team in team_metrics: 
      team_name = team['team_name']
      logging.info(f"Attempting to insert game logs for team '{team_name}")
      
      df = team['team_metrics']
      if df.empty:
         logging.warning(f"No team metrics corresponding to team '{team_name}; skipping insertion")
         continue
      
      # fetch team ID 
      team_id = get_team_id_by_name(team_name, teams_and_ids)
      
      # fetch team game logs 
      tuples = get_team_log_tuples(df, team_id, curr_year)
      
      # insert team game logs into db
      insert_data.insert_team_game_logs(tuples)
   
   

'''
Utility function to fetch team game log tuples to insert into our database 

Args: 
   df (pd.DataFrame): data frame to extract into tuples
   team_id (int): id corresponding to team to fetch game logs for 
   year (int): year we are fetching game logs for 

Returns:
   tuples (list): list of tuples to be directly inserted into our database
'''
def get_team_log_tuples(df: pd.DataFrame, team_id: int, year: int): 
   tuples = []
   for _, row in df.iterrows(): 
      game_log = (
         team_id,
         row['week'],
         row['day'],
         service_util.get_game_log_year(row['week'], year),
         row['rest_days'],
         row['home_team'],
         row['distance_traveled'],
         team_service.get_team_id_by_name(row['opp']),
         row['result'], 
         row['points_for'], 
         row['points_allowed'], 
         row['tot_yds'],
         row['pass_yds'],
         row['rush_yds'],
         row['opp_tot_yds'],
         row['opp_pass_yds'],
         row['opp_rush_yds']
      )
      tuples.append(game_log)
   
   return tuples
   
'''
Utility function to retrieve a teams ID based on their name 

Args:
   team_name (str): team name to retrieve ID for 
   teams_and_ids (list): list of dictionaries containing team names and ids 

Returns: 
   id (int): ID corresponding to team name
'''
def get_team_id_by_name(team_name: str, teams_and_ids: list): 
   team = next((team for team in teams_and_ids if team["name"] == team_name), None)
   return team['team_id'] if team else None


'''
Functionality to determine if a game log was persisted for a given week 

Args:
   game_log_pk (dict): PK to check is persisted in DB 

Returns:
   game_log (dict): None or persisted game log
'''
def is_game_log_persisted(game_log_pk: dict):
   game_log = fetch_data.fetch_team_game_log_by_pk(game_log_pk)
   
   if game_log == None:
      return False
   else:
      return True


'''
Functionality to retrieve all game logs for a particular season 

Args:
   team_id (int): team ID to fetch game logs for
   year (int): year to fetch game logs for 

Returns:
   game_logs (list): list of game logs corresponding to team
'''
def get_teams_game_logs_for_season(team_id: int, year: int):
   return fetch_data.fetch_all_teams_game_logs_for_season(team_id, year)

'''
Utility function to determine if a teams game log has previously been inserted 

Args: 
   team_metrics (list): list of dictionaries containing team's name & game logs 
   curr_year (int): current year
   teams_and_ids (list): list of dictionaries containing team's name & corresponding team ID 
   
Returns:
   None
'''
def remove_previously_inserted_game_logs(team_metrics, curr_year, teams_and_ids):
   team_metric_pks = []

   # generate pks for each team game log
   for team in team_metrics:
      df = team['team_metrics']
      if len(df) == 1:
         week = str(df.iloc[0]['week'])
         team_metric_pks.append({"team_id": get_team_id_by_name(team['team_name'], teams_and_ids),"week": week, "year": service_util.get_game_log_year(week, curr_year)})
   
   # check if this execution is for recent games or not 
   if len(team_metrics) != len(team_metric_pks):
      logging.info("Program execution is not for most recent games; skipping check for previously persisted team game logs")
      return
   
   # remove duplicate entires 
   index = 0 
   while index < len(team_metrics):
      if is_game_log_persisted(team_metric_pks[index]):
         del team_metrics[index]
         del team_metric_pks[index]
      else:
         logging.debug(f'Team game log corresponding to PK [{team_metric_pks[index]}] not persisted; inserting new game log')
         index +=1

'''
Functionality to calculate rankings (offense & defense) for a team

Args:
   curr_year (int): year to take into account when fetching rankings 

Returns 
   None 
'''
def calculate_team_rankings(curr_year: int):
   # fetch all teams 
   teams = team_service.get_all_teams()
   
   for team in teams: 
      # fetch team game logs 
      team_game_logs = get_teams_game_logs_for_season(team.get("team_id"), curr_year)
      
      
      
   
   '''
   a) For each team:
      1) fetch game logs for relevant season 
      2) accumulate metrics for offense 
            - calculate_rush_rank 
            - calculate_pass_rank 
      3) accumulate metrics for defense 
            - calculate_rush_rank
            - calculate_pass_rank


   '''

'''
Functionality to calculate the rankings of an offense based on their game logs for a given season 

Args:
   team_game_logs (list): list of game logs for a particular team 

Returns:
   off_rush_rank, off_pass_rank (tuple): rankings of the teams offense 
'''
def calculate_off_rankings(team_game_logs: list):
   for game_log in team_game_logs:
      # accumulate total passing yards 
      
      # accumulate total rushing yards 

'''
Functionality to calculate the rankings of a defense based on their game logs for a given season 

Args:
   team_game_logs (list): list of game logs for a particular team 

Returns:
   def_rush_rank, def_pass_rank (tuple): rankings of the teams offense 
'''
def calculate_def_rankings(team_game_logs: list):
   return None