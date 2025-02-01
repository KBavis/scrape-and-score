from config import props
import requests
from db import fetch_data, insert_data
import logging
import time

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

    players = fetch_data.fetch_players_active_in_specified_year(season)

    # iterate through each potential player
    for player in players:
        player_name = player['name']
        logging.info(f'Fetching player props for the player "{player_name}" for the {season} season')

        first_and_last = player_name.lower().split(" ")[:2]
        player_slug = "-".join(first_and_last)

        # iterate through each relevant week in specified season
        player_props = {"player_id": player['id'], "player_name": player_name}
        season_odds = []
        for week in range(1, max_week + 1):
            event_ids = fetch_event_ids_for_week(week, season)
            betting_odds = get_player_betting_odds(player_slug, event_ids, markets)
            
            if betting_odds == None:
                logging.info(f'No betting odds retrieved for player {player_name} for week {week} in {season} season')
                continue
            else:
                season_odds.append({"week": week, "week_odds": betting_odds})
        
        player_props.update({"season_odds": season_odds})    
        
        # insert season long props into db
        logging.info(f'Attempting to insert player props for player {player_name} for the {season} season...')  
        insert_data.insert_player_props(player_props, season)


"""
Generate proprer URL needed to fetch relevant player prop metrics for a given week, player, and season 

Args:
    player_name (str): player_slug to pass as a parameter to our request
    event_ids (str): all event_ids pertaining to the specified week 
    market_ids (str): all relevant market IDs to fetch odds for

Returns:
    odds (dict): players odds 
"""


def get_player_betting_odds(player_name: str, event_ids: str, market_ids: str):
    base_url = props.get_config("website.betting-pros.urls.historical-odds")
    parsed_url = (
        base_url.replace("{MARKET_IDS}", market_ids)
        .replace("{PLAYER_SLUG}", player_name)
        .replace("{EVENT_IDS}", event_ids)
    )  
    market_id_mapping = generate_market_id_mapping()

    # fetch initial data from first page
    data = get_data(parsed_url.replace("{PAGE}", str(1)))
    num_pages = determine_number_of_pages(data)
    
    # no data for this player in the specified week 
    if num_pages == 0:
        return None
    
    # account for odds on first page
    odds = get_odds(data, market_id_mapping)

    # loop through each possible page
    for page in range(2, num_pages + 1):
        data = get_data(parsed_url.replace("{PAGE}", str(page)))
        page_odds = get_odds(data, market_id_mapping)
        
        # account for additional odds available 
        odds.extend(page_odds)
    
    return odds


"""
Determine how many pages of data we should iterate through 

Args:
    data (dict): dictionary containing relevant player odds 

Returns:
    pages (int): # of pages to iterate through
"""


def determine_number_of_pages(data: dict):
    if data and data["_pagination"]:
        return data["_pagination"]["total_pages"]


"""
Generate mapping of a market ID to its respective betting prop name 

TODO: Move this to application.yaml

Args:   
    None

Returns:
    mapping (dict): mapping of market ID to betting prop name 
"""


def generate_market_id_mapping():
    return {
        71: "Player To Score The Last Touchdown",
        253: "Fantasy Points Over/Under",
        75: "Most Receiving Yards",
        105: "Receiving Yards Over/Under",
        104: "Receptions Over/Under",
        66: "First Touchdown Scorer",
        78: "Anytime Touchdown Scorer",
        107: "Rushing Yards Over/Under",
        106: "Rushing Attempts Over/Under",
        101: "Interception Over/Under",
        103: "Passing Yards Over/Under",
        333: "Passing Attempts Over/Under",
        102: "Passing Touchdowns Over/Under",
        76: "Most Rushing Yards",
        100: "Passing Completions Over/Under",
        73: "Most Passing Touchdowns",
        74: "Most Passing Yards",
    }


"""
Determine relevant odds based on API response 

Args:
    data (dict): relevant data from API response 
    market_ids (dict): mapping of a market ID to corresponding props

Returns:
    odds (dict): relevatn odds for player 
"""


def get_odds(data: dict, market_ids: dict):
    odds = []
    
    # loop through available offers
    offers = data['offers']
    for offer in offers:
        market_id = offer['market_id']
        prop_name = market_ids[market_id]
        
        # loop through selections (will only be multiple if over/under)
        selections = offer['selections']
        for selection in selections:
            
            # update w/ over/under label if needed
            if selection['label'] == 'Over' or selection['label'] == 'Under': 
                full_prop_name = f"{prop_name} ({selection['label']})"
            else:
                full_prop_name = prop_name
            
            # loop through available bookies & find the best line
            bookies = selection['books']
            best_found = False
            for book in bookies:
                # stop looping if we found best line 
                if best_found:
                    break
                
                lines = book['lines']
            
                # loop through available lines for a bookie 
                for line in lines:
                    # only account for best lines
                    if line['best'] == True: 
                        odds.append({"label": full_prop_name, "cost": line['cost'], "line": line['line']})
                        best_found = True
                        break
                    
    return odds


"""
Retrieve JSON data from specified endpoint 

Args:
    url (str): url to retrieve data from 

Returns:
    jsonData (dict): json data
"""


def get_data(url: str):
    time.sleep(.25) #TODO: Make this a config
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


