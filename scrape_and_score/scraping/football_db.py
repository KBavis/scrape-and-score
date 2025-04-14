import logging
from bs4 import BeautifulSoup
from . import util
from config import props

def scrape_historical(start_year: int, end_year: int): 
    """Scrape & persist player injuries for previous seasons throughout each potential week of the NFL season 

    Args:
        start_year (int): year to start scraping metrics for 
        end_year (int): year to stop scraping metrics for 
    """
    logging.info(f"Scraping & persisting player_injuries from the year {start_year} to the year {end_year}")
    
    player_name_id_mapping = {} 

    for season in range(start_year, end_year + 1): 
        for week in range(1, 19):
            html = get_html(week, season)
            soup = BeautifulSoup(html, "html.parser")
            

            player_injuries = parse_all_player_injuries(soup)

            #TODO: Implement me
            # generate_and_persist_player_injury_records(player_injuries, season, week, player_name_id_mapping)





def scrape_upcoming(week: int, season: int): 
    """Scrape & persist current player injuries in the upcoming week of the NFL season for players

    Args:
        week (int): week to retrieve injuries for 
        season (int): season to retrieve injuries for 
    """
    logging.info(f"Scraping & persisting player_injuries for week {week} for the {season} NFL season")
    html = get_html(week, season)



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
    wed_pract_status = practice_status_divs[0].get_text().strip().lower() if len(practice_status_divs) > 0 else None
    thurs_pract_status = practice_status_divs[1].get_text().strip().lower() if len(practice_status_divs) > 1 else None
    fri_pract_status = practice_status_divs[2].get_text().strip().lower() if len(practice_status_divs) > 2 else None

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




def generate_and_persist_player_injury_records(player_injuries: list, season: int, week: int, player_name_id_mapping: dict): 
    """Generate and persist player injury records into our datbase 

    Args:
        player_injuries (list): list of player names and their corresponding statuses 
        season (int): season these injuries correspond to 
        week (int): week these injuries correspond to 
        player_name_id_mapping (dict): mapping containing already mapped player normalized names & ids 
    """



def map_player_names_to_player_ids(player_names: list, player_name_id_mapping: dict): 
    """Map a player name to their respecitve player_id 

    Args:
        player_names (list): list of player_names to map to their ID
    """


def get_html(week: int, season: int):
    base_url = props.get_config('website.football-db.urls.player-injuries')
    return util.fetch_page(base_url.format(season, week))
