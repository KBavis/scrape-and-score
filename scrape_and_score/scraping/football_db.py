import logging
from bs4 import BeautifulSoup
from . import util
from config import props
from service import player_service
from db.read.players import (
    fetch_pks_for_inserted_player_injury_records
)
from db.insert.players import (
    update_player_injuries,
    insert_player_injuries
)

def scrape_historical(start_year: int, end_year: int): 
    """Scrape & persist player injuries for previous seasons throughout each potential week of the NFL season 

    Args:
        start_year (int): year to start scraping metrics for 
        end_year (int): year to stop scraping metrics for 
    """
    logging.info(f"Scraping & persisting player_injuries from the year {start_year} to the year {end_year}")

    for season in range(start_year, end_year + 1): 
        for week in range(1, 19):
            html = get_html(week, season)
            soup = BeautifulSoup(html, "html.parser")
            

            player_injuries = parse_all_player_injuries(soup)

            generate_and_persist_player_injury_records(player_injuries, season, week)
            

def scrape_upcoming(week: int, season: int, player_ids: list = None): 
    """Scrape & persist current player injuries in the upcoming week of the NFL season for players

    Args:
        week (int): week to retrieve injuries for 
        season (int): season to retrieve injuries for 
        player_ids (list): player IDs corresponding to teams that have yet to play their games 
    """
    logging.info(f"Scraping & persisting player_injuries for week {week} for the {season} NFL season")

    html = get_html(week, season)
    soup = BeautifulSoup(html, "html.parser")

    player_injuries = parse_all_player_injuries(soup)

    generate_and_persist_player_injury_records(player_injuries, season, week, player_ids)





def parse_all_player_injuries(soup: BeautifulSoup): 
    """Parse all relevant player injuries for each team listed

    Args:
        soup (BeautifulSoup): parsed HTML 
    """
    player_injuries = []

    team_injuries = soup.find_all("div", class_=["divtable", "divtable-striped", "divtable-mobile"])
    for team_injury_table in team_injuries: 
        player_injuries.extend(parse_team_injuries(team_injury_table))
    
    return player_injuries



def parse_team_injuries(team_injury_table: BeautifulSoup):
    """Parse all player injuries from HTML corresponding to a particular team

    Args:
        team_name (str): team name to scrape infomraiton for 
        soup (BeautifulSoup): parsed html 
    """
    player_injuries = []
    player_rows = team_injury_table.find_all("div", class_="tr")

    
    for player in player_rows: 
        player_injuries.append(extract_player_statuses_and_injury(player))
    
    return player_injuries
    
        
    
def extract_player_statuses_and_injury(player: BeautifulSoup): 
    """
    Extract player practice/game statuses and corresponding injury for a given week/season

    Args:
        player (BeautifulSoup): parsed HTML containing relevant player injury information

    Returns:
       dict: key-value mapping of relevant player injury information 
    """
    # extract player name 
    player_name = player.find("a").get_text().strip()

    # extract injury location(s)
    injury_location_div = player.find("div", class_="td w15 d-none d-md-table-cell")
    injury_locations = injury_location_div.get_text().strip().lower() if injury_location_div else None

    # extract practice statuses
    practice_status_divs = player.find_all("div", class_="td center w15 d-none d-md-table-cell")
    wed_pract_status = normalize_status(practice_status_divs[0].get_text()) if len(practice_status_divs) > 0 else None
    thurs_pract_status = normalize_status(practice_status_divs[1].get_text()) if len(practice_status_divs) > 1 else None
    fri_pract_status = normalize_status(practice_status_divs[2].get_text()) if len(practice_status_divs) > 2 else None

    # extract game status (last td.w20 div)
    game_status_divs = player.find_all("div", class_="td w20 d-none d-md-table-cell")
    game_status = game_status_divs[-1].get_text().strip() if game_status_divs else None


    return {
        "player_name": player_name, 
        "injury_locations": injury_locations,
        "wed_prac_sts": wed_pract_status, 
        "thurs_prac_sts": thurs_pract_status, 
        "fri_prac_sts": fri_pract_status,
        "off_sts": extract_game_status(game_status)
    }


def normalize_status(status): 
    """Helper function to normalize practice status

    Args:
        status (str): status to normalize

    Returns:
        str : normalized status 
    """
    status = status.strip().lower() 
    return None if status == '--' else status


def extract_game_status(text: str):
    """Extract the official game status of a player

    Args:
        text (str): game status parsed from HTML 

    Returns:
        str : game status 
    """
    text = text.strip()
    
    if text == "--":
        return None
    if ") " in text and " @" in text:
        try:
            after_date = text.split(") ")[1] 
            status = after_date.split(" @")[0].strip().lower()
            return status
        except IndexError:
            return None
    return None




def generate_and_persist_player_injury_records(player_injuries: list, season: int, week: int, player_ids: list, player_name_id_mapping: dict = {}): 
    """Generate and persist player injury records into our datbase 

    Args:
        player_injuries (list): list of player names and their corresponding statuses 
        season (int): season these injuries correspond to 
        week (int): week these injuries correspond to 
        player_name_id_mapping (dict): mapping containing already mapped player normalized names & ids 
    """

    # fetch previously persisted records in order to determine which records require updates vs insertion 
    persisted_player_injuries = fetch_pks_for_inserted_player_injury_records(season, week)

    persistable_records = []
    for record in player_injuries: 
        player_normalized_name = player_service.normalize_name(record["player_name"])

        # extract player ID by normalized name
        if player_normalized_name not in player_name_id_mapping:
            player_id = player_service.get_player_id_by_normalized_name(player_normalized_name)
            player_name_id_mapping[player_normalized_name] = player_id
        else:
            player_id = player_name_id_mapping[player_normalized_name]

        
        if player_id is None:
            logging.warn(f"Skipping record {record} for insertion since no player is persisted with normalized name {player_normalized_name}")
            continue


        # validate player ID is corresponding to player whom hasn't played yet 
        if player_ids is not None:
            if player_id not in player_ids:
                logging.warning(f"Skipping record {record} for insertion/update since player has already played their game.")
                continue
        
        # generate record 
        persistable_records.append(
            {
                "player_id": player_id, 
                "week": week, 
                "season": season, 
                "injury_loc": record['injury_locations'], 
                "wed_prac_sts": record['wed_prac_sts'], 
                "thurs_prac_sts": record['thurs_prac_sts'],
                "fri_prac_sts": record['fri_prac_sts'], 
                "off_sts": record['off_sts']
            }
        )
    
    updated_records, insert_records = filter_persisted_records(persistable_records, persisted_player_injuries)

    if updated_records:
        logging.info(f"Attemtping to update {len(updated_records)} player_injuries records in database.")
        update_player_injuries(updated_records)
    
    if insert_records:
        logging.info(f"Attemtping to insert {len(insert_records)} player_injuries records into database.")
        insert_player_injuries(insert_records)
        

        
def filter_persisted_records(records: list, record_pks: list):
    """
    Filter records by determining if they are previously persisted or not,
    ensuring no duplicate primary keys exist in the final update/insert lists.

    Args:
        records (list): list of new records we want to persist 
        record_pks (list): list of record pks that we will use to filter 
    """
    logging.info("Filtering player_injuries records to determine if they should be updated in database or inserted")

    seen_pks = set()
    records_to_insert = []
    records_to_update = []

    for record in records:
        pk = (record['player_id'], record['week'], record['season'])
        if pk in seen_pks:
            continue  # skip duplicate PKs
        seen_pks.add(pk)

        pk_dict = {"player_id": pk[0], "week": pk[1], "season": pk[2]}
        if pk_dict in record_pks:
            records_to_update.append(record)
        else:
            records_to_insert.append(record)

    return records_to_update, records_to_insert


def get_html(week: int, season: int):
    base_url = props.get_config('website.football-db.urls.player-injuries')
    return util.fetch_page(base_url.format(season, week))
