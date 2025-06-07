import logging
from scrape_and_score.db.read.teams import fetch_team_by_name, fetch_all_teams


def get_team_id_by_name(name: str):
    """
    Functionality to retrieve a teams ID by their name

    Args:
       name (str): name to fetch ID for

    Returns:
       id (int): ID corresponding to team name
    """

    logging.info(f"Fetching team ID for team corresponding to name '{name}'")
    team = fetch_team_by_name(name)

    if team == None:
        logging.warning(f"Unable to locate team by the name: {name}")
        return None

    return team["team_id"]


def get_all_teams():
    """
    Functionality to retrieve all teams persisted within our DB

    Args:
       None

    Returns:
       teams (list): list of dictionary items representing teams from our DB
    """

    logging.info(f"Fetching all teams persisted within our DB")
    teams = fetch_all_teams()
    return teams