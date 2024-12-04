from .connection import get_connection, close_connection
import logging 

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
      logging.error(f'An exception occured while inserting the following teams into our db: {teams}', e)
      raise e


'''
Functionality to persist a single team into our db 

Args: 
   team_name (str): team to insert into our db 

Returns: 
   team_id (int): id corresponding to a particular team 
'''
def insert_team(team_name: dict): 
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
      logging.error(f"An exception occured while inserting the following team '{team_name}' into our db", e)
      raise e
      
   
