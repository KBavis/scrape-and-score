from db import fetch_data
import logging
import re

"""
Functionality to retrieve all players persisted within our DB 
"""


def get_all_players():
    logging.info(f"Fetching all players persisted within our database")
    players = fetch_data.fetch_all_players()
    logging.info(f"Retrieved {len(players)} players from database")
    return players


def get_player_id_by_normalized_name(name: str):
    """Retrieve player ID corresponding to players normalized name

    Args:
        name (str): the name of the player to retrieve player ID for 
    """
    normalized_name= normalize_name(name)
    logging.info(f"Retrieving the player ID corresponding to the normalized name: {normalized_name}")
    player_id = fetch_data.fetch_player_id_by_normalized_name(normalized_name)
    return player_id

def get_player_id_by_position_season_and_normalized_name(season: int, position: int, name: str):
    """
    Retrieve a player ID corresponding to a particular season, normalized name, and position

    Args: 
        season (int): the relevant season 
        name (str): players normlaized name 
        position (str): relevant position
    """

    logging.info(f"Attempting to retrieve player ID for  player[normalized_name={name},position={position},season={season}]")
    player_id = fetch_data.fetch_player_id_by_normalized_name_season_and_position(name, position, season)
    return player_id


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z ]", "", name).lower()