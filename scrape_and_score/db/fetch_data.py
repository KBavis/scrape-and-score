import logging 
from .connection import get_connection, close_connection

'''
Functionality to fetch multiple teams 
'''
def fetch_all_teams():
   sql = "SELECT * FROM team"
   teams = []

   try:
      connection = get_connection()

      with connection.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            
            for row in rows:
               teams.append({
                  'team_id': row[0],
                  'name': row[1],
                  'offense_rank': row[2],
                  'defense_rank': row[3],
               })

   except Exception as e:
      logging.error("An error occurred while fetching all teams: {e}")
      raise e

   return teams

'''
Functionality to fetch a team by their team name 

Args:
   team_name (str): team name to retrieve team by

Returns:
   team (dict): team record
'''
def fetch_team_by_name(team_name: int):
   sql = "SELECT * FROM team WHERE name = %s"
   team = None  

   try:
      connection = get_connection()

      with connection.cursor() as cur:
            cur.execute(sql, (team_name,)) # ensure team_name in tuple 
            row = cur.fetchone()  
            
            if row:
               team = {
                  'team_id': row[0],
                  'name': row[1],
                  'offense_rank': row[2],
                  'defense_rank': row[3],
               }

   except Exception as e:
      logging.error(f"An error occurred while fetching team with name {team_name}: {e}")
      raise e


   return team


'''
Functionality to fetch all players

Returns:
   players (list): list of players persisted in DB
'''
def fetch_all_players():
   sql = "SELECT * FROM player"
   players = []

   try:
      connection = get_connection()

      with connection.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            
            for row in rows:
               players.append({
                  'player_id': row[0],
                  'team_id': row[1],
                  'player_name': row[2],
                  'position': row[3],
               })

   except Exception as e:
      logging.error(f"An error occurred while fetching all players: {e}")
      raise e

   return players

'''
Functionality to fetch a player by their player name 

Args:
   player_name (str): player name to retrieve player by

Returns:
   player (dict): player record
'''
def fetch_player_by_name(player_name: str):
   sql = "SELECT * FROM player WHERE name = %s"
   player = None  

   try:
      connection = get_connection()

      with connection.cursor() as cur:
            cur.execute(sql, (player_name,))  # ensure player_name is passed as a tuple
            row = cur.fetchone()  
            
            if row:
               player = {
                  'player_id': row[0],
                  'team_id': row[1],
                  'name': row[2],
                  'position': row[3],
               }

   except Exception as e:
      logging.error(f"An error occurred while fetching player with name {player_name}: {e}.")
      raise e

   return player