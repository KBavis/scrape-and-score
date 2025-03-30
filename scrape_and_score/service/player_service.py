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


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z ]", "", name).lower()