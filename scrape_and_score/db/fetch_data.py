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


'''
Functionality to retrieve a single team game log from our DB 

Args:
   None 
Returns: 
   player_game_log (dict): team game log or None if not found 
'''
def fetch_one_team_game_log():
   sql = 'SELECT * FROM team_game_log FETCH FIRST 1 ROW ONLY'
   team_game_log = None
   
   try:
      connection = get_connection()

      with connection.cursor() as cur:
            cur.execute(sql)  
            row = cur.fetchone()  
            
            if row:
               team_game_log = {
                  'team_id': row[0], 
                  'week': row[1],
                  'year': row[3]
               }

   except Exception as e:
      logging.error(f"An error occurred while fetching one record from team_game_log: {e}")
      raise e
   
   return team_game_log


'''
Functionality to retrieve a single player game log from our DB 

Args:
   None 
Returns: 
   player_game_log (dict): player game log or None if not found 
'''
def fetch_one_player_game_log():
   sql = 'SELECT * FROM player_game_log FETCH FIRST 1 ROW ONLY'
   player_game_log = None
   
   try:
      connection = get_connection()

      with connection.cursor() as cur:
            cur.execute(sql)  
            row = cur.fetchone()  
            
            if row:
               player_game_log = {
                  'player_id': row[0],
                  'week': row[1],
                  'year': row[3]
               }

   except Exception as e:
      logging.error(f"An error occurred while fetching one record from player_game_log: {e}")
      raise e
   
   return player_game_log



"""
Functionality to retrieve a team's game log by its PK (team_id, week, year)

Args:
   pk (dict): primary key for a given team's game log (team_id, week, year)

   Returns:
   team_game_log (dict): the team game log corresponding to the given PK 
"""
def fetch_team_game_log_by_pk(pk: dict):
    sql = 'SELECT * FROM team_game_log WHERE team_id=%s AND week=%s AND year=%s'
    team_game_log = None
    
    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (pk['team_id'], pk['week'], pk['year'])) 
            row = cur.fetchone()  
            
            if row:
               team_game_log = {
                  'team_id': row[0],
                  'week': row[1],
                  'day': row[2],
                  'year': row[3],
                  'rest_days': row[4],
                  'home_team': row[5],
                  'distance_traveled': row[6],
                  'opp': row[7],
                  'result': row[8],
                  'points_for': row[9],
                  'points_allowed': row[10],
                  'tot_yds': row[11],
                  'pass_yds': row[12],
                  'rush_yds': row[13],
                  'opp_tot_yds': row[14],
                  'opp_pass_yds': row[15],
                  'opp_rush_yds': row[16]
               }

    except Exception as e:
        logging.error(f"An error occurred while fetching the team game log corresponding to PK {pk}: {e}")
        raise e
    
    return team_game_log


'''
Functionality to retrieve a players game log by its PK (player_id, week, year)

Args:
   pk (dict): primary key for a given player's game log (player_id, week, year)
Returns:
   player_game_log (dict): the player game log corresponding to the given PK 
'''
def fetch_player_game_log_by_pk(pk: dict):
   sql = 'SELECT * FROM player_game_log WHERE player_id=%s AND week=%s AND year=%s'
   player_game_log = None
   
   try:
      connection = get_connection()

      with connection.cursor() as cur:
            cur.execute(sql, (pk['player_id'], pk['week'], pk['year']))  
            row = cur.fetchone()  
            
            if row:
               player_game_log = {
                  'player_id': row[0],
                  'week': row[1],
                  'day': row[2],
                  'year': row[3],
                  'home_team': row[4],
                  'opp': row[5],
                  'result': row[6], 
                  'points_for': row[7],
                  'points_allowed': row[8],
                  'completions': row[9],
                  'attempts': row[10],
                  'pass_yd': row[11], 
                  'pass_td': row[12],
                  'interceptions': row[13], 
                  'rating': row[14], 
                  'sacked': row[15], 
                  'rush_att': row[16], 
                  'rush_yds': row[17], 
                  'rush_tds': row[18], 
                  'tgt': row[19], 
                  'rec': row[20], 
                  'rec_yd': row[21], 
                  'rec_td': row[22], 
                  'snap_pct': row[23]
               }

   except Exception as e:
      logging.error(f"An error occurred while fetching the player game log corresponding to PK {pk}: {e}")
      raise e
   
   return player_game_log
   