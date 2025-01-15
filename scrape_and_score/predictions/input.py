from config import props 
from db import fetch_data

'''
Functionality to prompt user for player's name & upcoming matchup to configure our prediction 

Args:
    None 

Returns: 
    TODO
'''
def get_user_input():
    valid = False
    players_name = ''

    while not valid and players_name != 'exit':
        players_name = input("\n\n\nPlease enter the players full name (first & last) that you would like us to predict fantasy points for:\n")
        valid = validate_players_name(players_name)
        if not valid:
            print(f'The name {players_name} is not valid, please try a different name.')


'''
Determine if inputed player's name is valid or not (i.e we have persisted data on specified player)

Args:
    player_name (str): players name to validate 

Returns: 
    valid (bool): true or false indicating if players name is valid
'''
def validate_players_name(player_name):
    # ensure player_name not empty
    if not player_name:
        return False
    
    # ensure not just first name passed in
    split_name = player_name.split() 
    if len(split_name) < 2:
        return False

    player = fetch_data.fetch_player_by_name(player_name.title())
    if player:
        return True 
    
    return False