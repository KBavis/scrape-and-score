import logging
from db import fetch_data

'''
Functionality to retrieve a teams ID by their name 

Args:
   name (str): name to fetch ID for 

Returns:
   id (int): ID corresponding to team name 
'''
def get_team_id_by_name(name: str):
   logging.info(f"Fetching team ID for team corresponding to name '{name}'")
   team = fetch_data.fetch_team_by_name(name)
   return team['team_id']