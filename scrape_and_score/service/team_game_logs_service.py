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
         week = df.iloc[0]['week']
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