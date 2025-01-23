from .connection import get_connection, close_connection
import logging 
import pandas as pd

'''
Functionality to persist a particular player 

Args: 
   players (list): list of players to persist 
'''
def insert_players(players: list): 
   try: 
      for player in players:
         insert_player(player)
         
   except Exception as e: 
      logging.error(f'An exception occurred while attempting to insert players {players}', exc_info=True)
      raise e

'''
Functionality to persist a single player 

Args: 
   player (dict): player to insert into our db 
   team_id (int): ID of the team corresponding to the player
'''
def insert_player(player: dict): 
   query = '''
      INSERT INTO player (team_id, name, position) 
      VALUES (%s, %s, %s)
   '''
   
   try: 
      # ensure that team_id and player fields are correctly passed into the query
      player_name = player['player_name']
      player_position = player['position']
      team_id = player['team_id']
      
      # fetch connection to the DB
      connection = get_connection()

      with connection.cursor() as cur: 
         cur.execute(query, (team_id, player_name, player_position))  # Pass parameters as a tuple
         
         # Commit the transaction to persist data
         connection.commit()
         logging.info(f"Successfully inserted player {player_name} into the database")
      
   except Exception as e: 
      logging.error(f"An exception occurred while inserting the player {player}", exc_info=True)
      raise e

'''
Functionality to persist mutliple teams into our db 

Args: 
   teams (list): list of teams to insert into our db 
Returns: 
   teams (list): mapping of team names and ids we inserted

''' 
def insert_teams(teams: list):
   team_mappings = []
   
   try: 
      for team in teams: 
         team_id = insert_team(team)
         team_mappings.append({'team_id': team_id, 'name': team})
      
      return team_mappings
         
   except Exception as e:
      logging.error(f'An exception occured while inserting the following teams into our db: {teams}: {e}')
      raise e


'''
Functionality to persist a single team into our db 

Args: 
   team_name (str): team to insert into our db 

Returns: 
   team_id (int): id corresponding to a particular team 
'''
def insert_team(team_name: str): 
   sql = "INSERT INTO team (name) VALUES (%s) RETURNING team_id"
   
   try: 
      connection = get_connection()
      
      with connection.cursor() as cur: 
         cur.execute(sql, (team_name,)) # ensure team name is a tuple
         
         rows = cur.fetchone() 
         if rows:
            team_id = rows[0]
         
         connection.commit()
         return team_id
      
   except Exception as e: 
      logging.error(f"An exception occured while inserting the following team '{team_name}' into our db: {e}")
      raise e
      

'''
Functionality to persist multiple game logs for a team 

Args:
   game_logs (list): list of tuples to insert into team_game_logs 
   
Returns:
   None
'''
def insert_team_game_logs(game_logs: list):
   sql = '''
      INSERT INTO team_game_log (team_id, week, day, year, rest_days, home_team, distance_traveled, opp, result, points_for, points_allowed, tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
   '''
   
   try:
      connection = get_connection()

      with connection.cursor() as cur:
         cur.executemany(sql, game_logs)
         
         connection.commit()
         logging.info(f"Successfully inserted {len(game_logs)} team game logs into the database")

   except Exception as e:
      logging.error(f"An exception occurred while inserting the team game logs: {game_logs}", exc_info=True)
      raise e


'''
Functionality to persist multiple game logs for a RB 

Args:
   game_logs (list): list of tuples to insert into player_game_logs 

Returns: 
   None
'''
def insert_rb_player_game_logs(game_logs: list): 
   sql = '''
      INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, rush_att, rush_yds, rush_tds, tgt, rec, rec_yd, rec_td)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
   '''
   
   try:
      connection = get_connection()

      with connection.cursor() as cur:
         cur.executemany(sql, game_logs)
         
         connection.commit()
         logging.info(f"Successfully inserted {len(game_logs)} RB game logs into the database")

   except Exception as e:
      logging.error(f"An exception occurred while inserting the RB game logs: {game_logs}", exc_info=True)
      raise e


'''
Functionality to persist multiple game logs for a QB

Args:
   game_logs (list): list of tuples to insert into player_game_logs 

Returns: 
   None
'''
def insert_qb_player_game_logs(game_logs: list): 
   sql = '''
         INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, completions, attempts, pass_yd, pass_td, interceptions, rating, sacked, rush_att, rush_yds, rush_tds)
         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
   '''
   
   try:
      connection = get_connection()

      with connection.cursor() as cur:
         cur.executemany(sql, game_logs)
         
         connection.commit()
         logging.info(f"Successfully inserted {len(game_logs)} QB game logs into the database")

   except Exception as e:
      logging.error(f"An exception occurred while inserting the QB game logs: {game_logs}", exc_info=True)
      raise e

   
'''
Functionality to persist multiple game logs for a WR or TE

Args:
   game_logs (list): list of tuples to insert into player_game_logs 

Returns: 
   None
'''
def insert_wr_or_te_player_game_logs(game_logs: list): 
   sql = '''
      INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, tgt, rec, rec_yd, rec_td, snap_pct)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
   '''
   
   try:
      connection = get_connection()

      with connection.cursor() as cur:
         cur.executemany(sql, game_logs)
         
         connection.commit()
         logging.info(f"Successfully inserted {len(game_logs)} WR/TE game logs into the database")

   except Exception as e:
      logging.error(f"An exception occurred while inserting the WR/TE game logs: {game_logs}", exc_info=True)
      raise e

   

'''
Functionality to update player game log with calculated fantasy points

Args:
   fantasy_points (list): list of items containing player_game_log PK and their fantasy points

Returns: 
   None
'''
def add_fantasy_points(fantasy_points: list): 
   sql = 'UPDATE player_game_log SET fantasy_points = %s WHERE player_id = %s AND week = %s AND year = %s'
   
   try:
      connection = get_connection()
      
      params = [(log["fantasy_points"], log["player_id"], log["week"], log["year"]) for log in fantasy_points] # create needed tuples for executemany functionality

      with connection.cursor() as cur:
         cur.executemany(sql, params)
         connection.commit()
         logging.info(f"Successfully inserted {len(fantasy_points)} players fantasy points into the database")

   except Exception as e:
      logging.error(f"An exception occurred while inserting the follwoing fantasy points: {fantasy_points}", exc_info=True)
      raise e


'''
Update 'team' record with newly determined rankings 

Args:
   rankings (list): list of rankings to persist 
   col (str): column in 'team' to update based on rankings
'''
def update_team_rankings(rankings: list, col: str):
   sql = f'UPDATE team SET {col} = %s WHERE team_id = %s'
   
   try:
      connection = get_connection()
      
      params = [(rank['rank'], rank['team_id']) for rank in rankings] # create tuple needed to update records

      with connection.cursor() as cur:
         cur.executemany(sql, params)
         connection.commit()
         logging.info(f"Successfully updated {len(rankings)} {col} rankings in the database")

   except Exception as e:
      logging.error(f"An exception occurred while inserting the following rankings: {rankings}", exc_info=True)
      raise e


'''
Functionality to insert historical betting odds into our DB 

Args:
   betting_odds (list): list of historical betting odds 

Returns:
   None
'''
def insert_teams_odds(betting_odds: list, upcoming=False):
   if not upcoming:
      sql = '''
         INSERT INTO team_betting_odds (home_team_id, away_team_id, home_team_score, away_team_score, week, year, game_over_under, favorite_team_id, spread, total_points, over_hit, under_hit, favorite_covered, underdog_covered)
         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
      '''
      params = [(odds['home_team_id'], odds['away_team_id'], odds['home_team_score'], odds['away_team_score'], odds['week'], odds['year'], odds['game_over_under'], odds['favorite_team_id'], odds['spread'], odds['total_points'], odds['over_hit'], odds['under_hit'], odds['favorite_covered'], odds['underdog_covered']) for odds in betting_odds]
   else:
      sql = '''
         INSERT INTO team_betting_odds (home_team_id, away_team_id, week, year, game_over_under, favorite_team_id, spread)
         VALUES (%s, %s, %s, %s, %s, %s, %s)
      '''
      params = [(odds['home_team_id'], odds['away_team_id'], odds['week'], odds['year'], odds['game_over_under'], odds['favorite_team_id'], odds['spread']) for odds in betting_odds]
   
   try:
      connection = get_connection()

      with connection.cursor() as cur:
         cur.executemany(sql, params)
         connection.commit()
         logging.info(f"Successfully insert {len(betting_odds)} team {"historical" if not upcoming else "upcoming"} odds into our database")

   except Exception as e:
      logging.error(f"An exception occurred while inserting the following  {"historical" if not upcoming else "upcoming"} team odds: {betting_odds}", exc_info=True)
      raise e