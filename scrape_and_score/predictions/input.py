from config import props 
from db import fetch_data

'''
Functionality to prompt user for player's name & upcoming matchup to configure our prediction 

Args:
    None 

Returns: 
    opp_name, player_name (tuple): validated opponent name and player name
'''
def get_user_input():
    name_valid = False
    opp_valid = False
    
    players_name = ''
    opp_name = ''
    
    nfl_teams = props.get_config('nfl.teams')
    teams = [team['name'] for team in nfl_teams]

    #TODO: Make user input week/year combo and player on one of the teams (validate this is the case). Then we can check if we have entry in team_betting_odds corresponding to this

    while not name_valid and players_name != 'exit':
        players_name = input("\n\n\nPlease enter the players full name (first & last) that you would like us to predict fantasy points for:\n")
        name_valid = validate_players_name(players_name)
        if not name_valid:
            print(f'The name {players_name} is not valid, please try a different name.')
    
    if(players_name == 'exit'):
        exit(-1)
    
    while not opp_valid and opp_name != 'exit':
        opp_name = input("\nPlease enter the opposing teams full name (i.e Baltimore Ravens, New York Giants, etc) that you your player is going up against:\n")
        opp_valid = validate_nfl_team_name(teams, opp_name)
        if not opp_valid:
            print(f'The NFL team name {opp_name} is not valid, please try a different NFL team name.')

    return opp_name.title(), players_name.title()

'''
Functionaltiy to validate an NFL team name passed in 

Args:
    teams (list): valid NFL team name 
    team_name (str): name inputed by user 

Returns:
    valid (bool): whether name is valid or not
'''
def validate_nfl_team_name(teams: list, team_name: str):
    if not team_name:
        return False
    
    capitalized_team_name = team_name.title()
    
    if capitalized_team_name not in teams:
        return False 
    return True

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