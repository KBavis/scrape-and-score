from config import props
from db import fetch_data

"""
Functionality to prompt user for player's name & upcoming matchup to configure our prediction 

Args:
    None 

Returns: 
    week, player_name (tuple): validated week and player name
"""


def get_user_input():
    name_valid = False
    players_name = ""

    week = ""
    week_valid = False

    year = props.get_config("nfl.current-year")

    while not name_valid and players_name != "exit":
        players_name = input(
            "\n\n\nPlease enter the players full name (first & last) that you would like us to predict fantasy points for:\n"
        )
        name_valid = validate_players_name(players_name)
        if not name_valid:
            print(f"The name {players_name} is not valid, please try a different name.")

    if players_name == "exit":
        exit(-1)

    while not week_valid and week != "exit":
        week = input(
            "\nPlease enter the week that your players matchup is taking place:\n"
        )
        week_valid = validate_week_and_player_name(players_name, week, year)
        if not week_valid:
            print(
                f"The week {week} is not valid. Either you specified an invalid week, or no matchup regarding your specified player & this week is taking place. Please try a different week."
            )

    return week, players_name.title()


"""
Functionality to validate that the specified week has a matchup taking place with the specified player 

Args:
    player_name (str): the players name to ensure is playing in the given week
    week (str): the week of the matchup we need to account for 
    year (int): year to account for 

Returns:
    valid (bool): validity of the week given the players name
"""


def validate_week_and_player_name(player_name: str, week: str, year: int):
    # ensure week is numeric
    try:
        week_casted = int(week)
    except ValueError:
        return False

    # ensure player has matchup in given week
    valid = fetch_data.validate_week_and_corresponding_player_entry_exists(
        player_name, week_casted, year
    )
    return valid


"""
Determine if inputed player's name is valid or not (i.e we have persisted data on specified player)

Args:
    player_name (str): players name to validate 

Returns: 
    valid (bool): true or false indicating if players name is valid
"""


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
