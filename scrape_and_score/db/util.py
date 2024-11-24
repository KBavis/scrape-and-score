from .fetch_data import fetch_all_teams, fetch_all_players

'''
Functionality to determine if there are no entries in our team table 

Returns:
   flag (bool): flag to indicate if team is empty
'''
def is_team_empty():
   teams = fetch_all_teams() 
   
   if teams is None or len(teams) == 0:
      return True 
   return False


'''
Functionality to determine if there are no entries in our player table 

Returns:
   flag (bool): flag to indicate if player is empty
'''
def is_player_empty(): 
   players = fetch_all_players() 
   
   if players is None or len(players) == 0:
      return True 
   return False
   
   