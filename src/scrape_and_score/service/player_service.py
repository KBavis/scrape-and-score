import logging
import re
from scrape_and_score.db.read.players import (
    fetch_player_id_by_normalized_name,
    fetch_player_name_by_id,
    fetch_player_id_by_normalized_name_season_and_position,
)


def get_player_id_by_normalized_name(name: str):
    """Retrieve player ID corresponding to players normalized name

    Args:
        name (str): the name of the player to retrieve player ID for
    """

    normalized_name = normalize_name(name)
    logging.info(
        f"Retrieving the player ID corresponding to the normalized name: {normalized_name}"
    )
    player_id = fetch_player_id_by_normalized_name(normalized_name)
    return player_id


def get_player_name_by_id(id: int):
    """
    Fetch players name by their ID

    Args:
        id (int): player ID to retireve name for

    Returns:
        str: player's name
    """

    return fetch_player_name_by_id(id)


def get_player_id_by_position_season_and_normalized_name(
    season: int, position: int, name: str
):
    """
    Retrieve a player ID corresponding to a particular season, normalized name, and position

    Args:
        season (int): the relevant season
        name (str): players normlaized name
        position (str): relevant position
    """

    logging.info(
        f"Attempting to retrieve player ID for  player[normalized_name={name},position={position},season={season}]"
    )
    player_id = fetch_player_id_by_normalized_name_season_and_position(
        name, position, season
    )
    return player_id


def normalize_name(name: str) -> str:
    """
    Normalize a players name in the same manner that we apply to 'normalized_name' in our DB

    Args:
        name (str): name to normalize

    Returns:
        str: normalized player name
    """

    return re.sub(r"[^a-zA-Z ]", "", name).lower()
