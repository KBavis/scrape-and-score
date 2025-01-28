from db import fetch_all_players
import logging

"""
Functionality to retrieve all players persisted within our DB 
"""


def get_all_players():
    logging.info(f"Fetching all players persisted within our database")
    players = fetch_all_players()
    logging.info(f"Retrieved {len(players)} players from database")
    return players
