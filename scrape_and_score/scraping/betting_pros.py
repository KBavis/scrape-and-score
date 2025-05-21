from config import props
import requests
from service import player_service
from db import fetch_data, insert_data
import logging
import time

"""
Fetch all historical player odds 

TODO: superrrrr slow, need to add some concurrency to this

Args:
    season (int): season to retrieve player odds for 

Returns:   
    None
"""

def fetch_historical_odds(season: int):
    max_week = fetch_data.fetch_max_week_persisted_in_team_betting_odds_table(season) 
    markets = props.get_config("website.betting-pros.market-ids")

    #TODO: Only remove 253 market ID in the case that the game has already been played  or else it will throw exception when making request (we can add when game is upcoming)
    if markets.endswith(":253"):
        markets = markets[:-4] # remove last 4 occurence 

    players = fetch_data.fetch_players_active_in_specified_year(season) 


    # iterate through each potential player
    #TODO: account for najee harris slug being 'najee-harris-rb'
    #TODO: get metrics for amari cooper, tim patrick, samjae perine, noah brown, jahan dotson, hassan haskins, peyton hendershot, calvin ridley, jack stoll, jaelon darden, laviska shenault
    for player in players:
        player_name = player['name']
        logging.info(f'Fetching player props for the player "{player_name}" for the {season} season')

        first_and_last = player_name.replace("'", "").replace(".", "").lower().split(" ")[:2] 
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

        # insert season long player props into db 
        if season_odds:
            logging.info(f'Attempting to insert player props for player {player_name} for the {season} season...')  
            insert_data.insert_player_props(player_props, season)
        else:
            logging.warn(f"No player props found for player {player_name} and season {season}; skipping insertion")



def fetch_upcoming_player_odds_and_game_conditions(week: int, season: int, player_ids: list):

    # fetch & persist player odds 
    fetch_upcoming_player_odds(week, season, player_ids)

    # fetch & persist game conditions 
    fetch_upcoming_game_conditions(week, season)




def fetch_upcoming_game_conditions(week: int, season: int):
    """
    Retrieve upcoming game conditions & persist updates 

    Args:
        week (int): relevant week
        season (int): relevant season 
    """

    # extract games & their respective conditions for specific week/season 
    url = props.get_config("website.betting-pros.urls.events")
    parsed_url = url.replace("{WEEK}", str(week)).replace("{YEAR}", str(season))
    json = get_data(parsed_url)

    # iterate through each event
    for event in json["events"]:
        
        surface = extract_surface(event['venue'])
        
        weather = event['weather']
        temperature = weather['forecast_temp']
        wind_bearing = weather['forecast_wind_degree']
        precip_prob = weather['forecast_rain_chance']

        #TODO: Finsih me 



def extract_surface(venue: dict): 
    """
    Helper function to normalize the surface we are persisting 

    TODO: Split up logic to have surface_type and stadium_type (i.e Dome or Outdoors vs Artificial or Turf or Grass)

    TODO: Create mapping of venue stadium names to type of turf since I believe there are some incorrect values 

    Args:
        venue (dict): extracted venue from response 

    Returns:
        str: normalized surface
    """ 

    if not venue:
        logging.warning(f"No 'venue' retrieved from GET:/events response")
        return ""
    
    if venue['stadium_type'] == 'retractable_dome':
        return "Dome"
    
    if venue['surface'] == "turf" or venue['surface'] == 'artificial': 
        return "Turf"
    
    return venue['surface'].title() if venue['surface'] else ''


def fetch_upcoming_player_odds(week:int, season: int, player_ids: list):
    """
    Fetch relevant player props for upcoming NFL games & insert/update records into our DB

    TODO: Implement some sort of retry functionality for failed requests

    Args:
        week (int): relevant week 
        season (int): relevant season 
        player_ids (list): relevant player IDs we need to fetch odds for 

    """
    markets = props.get_config("website.betting-pros.market-ids")

    # extract event ids corresponding to week / season 
    event_ids = fetch_event_ids_for_week(week, season)

    records = []

    # iterate through each potential player
    for player_id in player_ids:

        # extract player name by ID 
        player_name = player_service.get_player_name_by_id(player_id)
        if player_name is None:
            logging.warning(f"Failed to retrieve player name corresponding to player ID {player_id}")
            continue

        logging.info(f'Fetching player props for the player "{player_name}" for week {week} of the {season} NFL season')

        # extract player slug for request 
        first_and_last = player_name.replace("'", "").replace(".", "").lower().split(" ")[:2] 
        player_slug = "-".join(first_and_last)

        # extract betting odds 
        betting_odds = get_player_betting_odds(player_slug, event_ids, markets)
            
        if betting_odds == None:
            logging.info(f'No betting odds retrieved for player {player_name} for week {week} in {season} season')
            continue
        
        records.append({"week": week, "season": season, "player_id": player_id, "player_name": player_name, "odds": betting_odds})

    update_records, insert_records = filter_upcoming_player_odds(records)

    # insert player betting odds
    if insert_records:
        logging.info(f'Attempting to insert {len(insert_records)} player_betting_odds records for week {week} of the {season} NFL season')  
        insert_data.insert_upcoming_player_props(insert_records)
    else:
        logging.warning(f"No new player props retrieved for week {week} of the {season} NFL season; skipping insertion")

    # update player betting odds 
    if update_records:
        logging.info(f'Attempting to update {len(update_records)}) player_betting_odds records for week {week} of the {season} NFL season')  
        insert_data.update_upcoming_player_props(update_records)
    else:
        logging.warning(f"No lines/costs modifications for players betting lines for week {week} of the {season} NFL season; skipping updates")
    

def filter_upcoming_player_odds(records: list):
    """
    Filter upcoming 'player_betting_odds' records to either be an insertable record or to be updated 

    Args:
        records (list): list of records retrieved for players 

    Returns:
        tuple: updateable & insertable records 
    """
    logging.info(f"Attempting to filter upcoming 'player_betting_odds' into insertable records vs updateable records")

    update_records = [] 
    insert_records = []

    # iterate through each indivudal player record
    for record in records:

        # extract relevant information
        player_id = record['player_id']
        week = record['week']
        season = record['season']
        odds = record['odds']

        # iterate through all available odds and filter
        for current_odds in odds:
            # extract label associated with current odds
            label = current_odds['label']

            # generate current record corresponding to current odds 
            curr_record = {"player_id": player_id, "player_name": record['player_name'],\
                            "week": week, "season": season, "label": label, \
                            "line": current_odds["line"], "cost": current_odds["cost"]}

            # retrieve player betting odds record by PK 
            persisted_record = fetch_data.fetch_player_betting_odds_record_by_pk(player_id, week, season, label)
            if persisted_record is None:
                logging.info(f"No record persisted for PK(player_id={player_id},week={week},season={season},label={label}); appending as insertable record")
                insert_records.append(curr_record)
                continue

            # check if modifications were made to record 
            if are_odds_modified(persisted_record, curr_record):
                logging.info(f"Player '{record['player_name']}' player_betting_odds with PK(player_id={player_id},week={week},season={season},label={label}) has been modified; appending as updatable record")
                update_records.append(curr_record)
                continue

    return update_records, insert_records


def are_odds_modified(persisted_record: dict, current_record: dict) -> bool:
    """
    Determines if the cost or line has changed for a player's odds record.

    Args:
        persisted_record (dict): The existing record from the DB
        current_record (dict): The new scraped record

    Returns:
        bool: True if either cost or line has changed; otherwise False.
    """

    # Check if cost or line has changed
    return (
        current_record["cost"] != persisted_record["cost"] or
        current_record["line"] != persisted_record["line"]
    )



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
    
    if not data:
        return None
    
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
    

    # remove duplicates by keeping the one with the highest cost
    seen_labels = {}
    for odd in odds:
        label = odd['label']
        if label in seen_labels:
            # if the label exists, compare costs and keep the higher cost
            if seen_labels[label]['cost'] < odd['cost']:
                seen_labels[label] = odd
        else:
            seen_labels[label] = odd

    # return the filtered list of odds with duplicates removed
    filtered_odds = list(seen_labels.values())
    return filtered_odds


"""
Determine how many pages of data we should iterate through 

Args:
    data (dict): dictionary containing relevant player odds 

Returns:
    pages (int): # of pages to iterate through
"""


def determine_number_of_pages(data: dict):
    if data and data.get("_pagination"):
        return data["_pagination"].get("total_pages", 0)
    return 0


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
            
            # skip under lines, only account for overs TODO: Remove me if needed 
            if selection['label'] == 'Under':
                continue
            
            # TODO: Remove me (no logner account for (over) / (under) as all lines used will simply be the over)
            if selection['label'] == 'Over': 
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
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        jsonData = response.json()
    except requests.RequestException as e:
        print(f"Error while fetching JSON content from the following URL {url} : {e}")
        return None
    
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


