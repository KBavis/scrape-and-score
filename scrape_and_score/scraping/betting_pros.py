from config import props
import requests
from db import fetch_data

"""
Fetch all historical player odds 

Args:
    season (int): season to retrieve player odds for 

Returns:   
    None
"""


def fetch_historical_odds(season: int):
    max_week = fetch_data.fetch_max_week_persisted_in_team_betting_odds_table(season)
    markets = props.get_config("website.betting-pros.market-ids")
    url = props.get_config("website.betting-pros.urls.historical-odds")

    for week in range(1, max_week + 1):
        event_ids = fetch_event_ids_for_week(week, season)


"""
Retrieve JSON data from specified endpoint 

Args:
    url (str): url to retrieve data from 

Returns:
    jsonData (dict): json data
"""


def get_data(url: str):
    headers = {"x-api-key": props.get_config("website.betting-pros.api-key")}
    jsonData = requests.get(url, headers=headers).json()
    return jsonData


"""
Retrieve event IDs (also known as game ID) for a given week/season

Args:   
    week (int): week to fetch ids for 
    year (int): year to fetch ids for 

Returns:
    ids (str): delimited ids corresponding to particular week/year
"""


def fetch_event_ids_for_week(week: int, year: int):
    url = props.get_config("website.betting-pros.urls.events")
    parsed_url = url.replace("{WEEK}", str(week)).replace("{YEAR}", str(year))

    json = get_data(parsed_url)

    ids = ":".join(str(event["id"]) for event in json["events"])
    return ids


"""
Functionality to generate a player slug 
"""


def generate_player_slug(player_name: str):
    return None
