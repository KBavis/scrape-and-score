"""
Module for retreving hisorical depth charts from OurLads 
"""
from . import util
from bs4 import BeautifulSoup
from config import props
import logging
from service import team_service, player_service
from datetime import datetime
from db import fetch_data, insert_data
import re

"""
Main entry point for scraping & persisting depth chart information across several years from OurLads 

Args:
    start_year (int): year to start scraping data from
    end_year (int): year to stop scraping data from 
Returns:
    None
"""
def scrape_and_persist(start_year: int, end_year: int): 
    # extract relevant teams 
    teams = [ 
        {
            "team": team["name"], 
            "acronym": team.get("our_lads_acronym", team["pfr_acronym"]) 
        } 
        for team in props.get_config("nfl.teams")
    ]
    logging.info(f"\nScraping & persisting depth charts for the seasons {start_year} - {end_year} and for the following teams: {teams}")

    # retrieve first page to extract relevant archive date IDs
    html = util.fetch_page("https://www.ourlads.com/nfldepthcharts/archive/150/IND")
    
    # parse html
    soup = BeautifulSoup(html, "html.parser")

    # extract archive datas 
    archive_dates_soup = soup.find_all("option")
    archive_dates = extract_archive_dates(archive_dates_soup, start_year, end_year)

    generate_player_and_player_teams_records(teams, start_year, end_year, archive_dates)

                 

def generate_player_and_player_teams_records(teams: list, start_year: int, end_year: int, archive_dates: list): 
    """
    Loop through all available seasons, for each potential team, and generate/persist relevant player & player_teams records 

    Args:
        teams (list): list of relevant team names and acronyms 
        start_year (int): the year to starting fetching data for
        end_year (int): the year to stop fetching data for
        archive_dates (list): mappings of a year to its corresponding archive IDs 
    
    """
    
    # loop through each potential team
    for team in teams: 
        team_id = team_service.get_team_id_by_name(team["team"])

        # cache player name & id mappings
        player_name_id_mapping = {}

        # loop through relevant seasons
        for season in range(start_year, end_year + 1): 
            logging.info(f"Attempting to determine depth charts for season {season} and for team {team['team']}")

            # extract archives dictionary corresponding to current season
            seasonal_archives_dict = next(item for item in archive_dates if item["season"] == season)
            archives = seasonal_archives_dict["archives"]

            # create date-week mapping 
            date_week_mapping = create_date_week_mapping(season)

            relevant_players = set() # set containing players unique to following season 
            start_date_mapping = {} # mapping of a players name to their start date
            end_date_mapping = {} # mapping of a player name to their end date 
            unique_player_records = [] # unique player records to persist 
            previous_dt = None
            player_depth_chart_position_records = [] # list of player depth chart position records 
            
            # loop through relevant months in current season (from Sept - Jan)
            for date, id in reversed(archives.items()): 
                
                # ensure current date referecnes the first of the month
                date = re.sub(r"/\d{2}/", "/01/", date)

                # retrieve depth chart for given month/year 
                html = util.fetch_page(f"https://www.ourlads.com/nfldepthcharts/archive/{id}/{(team['acronym'])}")
                soup = BeautifulSoup(html, "html.parser")

                 # extract depth chart
                depth_chart_table = soup.find("table")
                depth_chart = depth_chart_table.find("tbody")

                # extract fantasy relevant players 
                players = extract_fantasy_relevant_players(depth_chart)

                # generate player depth chart position records 
                generate_player_depth_chart_positions(players, date, date_week_mapping, player_depth_chart_position_records, season)

                # account for unique player records 
                add_unique_player_records(unique_player_records, players)

                # start & end date players 
                start_and_end_date_players_on_team(relevant_players, players, start_date_mapping, end_date_mapping, date_week_mapping, date, previous_dt, season)

                # account for previous date 
                previous_dt = date   

            # insert any new player records 
            insert_player_records(unique_player_records, team, player_name_id_mapping, season)

            # update player name to id mapping 
            for player_name in relevant_players:
                if player_name not in player_name_id_mapping:
                    normalized_name = player_service.normalize_name(player_name)
                    player = fetch_data.fetch_player_by_normalized_name(normalized_name)
                    player_name_id_mapping[player_name] = player['player_id']


            insert_player_teams_records(relevant_players, start_date_mapping, end_date_mapping, team_id, season, team, player_name_id_mapping)
            insert_player_depth_chart_position_records(player_name_id_mapping, player_depth_chart_position_records, season)
                    

def insert_player_depth_chart_position_records(player_name_id_mapping: dict, player_depth_chart_position_records: list, season: int):
    """
    Functionality to insert player depth chart position records into our database 

    Args:
        player_name_id_mapping (dict): mapping of a player name to an ID 
        player_depth_chart_position_records (list): list of relevant records to insert into our database 
        season (int): relevatn season this corresponds to 
    
    Returns: None
    """
    logging.info("Attempting to insert player depth chart position records")

    # update records with corresponding player ID 
    for record in player_depth_chart_position_records: 
        record["player_id"] = player_name_id_mapping[record["name"]]
        record["season"] = season
    
    # ensure record is unique (i.e not already inserted, and not a duplicate)
    filtered_depth_chart_records = []
    seen_records = set()
    for record in player_depth_chart_position_records:
        if fetch_data.fetch_player_depth_chart_record_by_pk(record) is None:
            record_key = (record["player_id"], record["week"], record["season"])

            if record_key not in seen_records:
                seen_records.add(record_key)
                filtered_depth_chart_records.append(record)

    if not filtered_depth_chart_records:
        logging.info(f"No new player depth chart records in the {season} season; skipping insertion")
    else:
        insert_data.insert_player_depth_charts(filtered_depth_chart_records)


def generate_player_depth_chart_positions(players: list, date: str, date_week_mapping: dict, player_depth_chart_position_records: list, season: int):
    """
    Functionality to generate relevant player depth chart records 

    Args:
        player (list): list of players that correspond to a depth chart position 
        date (str): date these players were retrieved from 
        date_week_mapping (dict): mapping of a date to a particular start and end date
        player_depth_chart_position_records: list of player depth chart positions 
        season (int): relevant season this depth chart corresponds to 
    
    Returns:    
        None
    """
    weeks = date_week_mapping.get(date)
    start_week = weeks["strt_wk"]
    end_week = weeks["end_wk"]

    for player in players:
        player_name = player["name"]
        depth_chart_pos = player["depth_chart_pos"]

        # generate depth chart position records 
        for week in range(start_week, end_week + 1):
            player_depth_chart_position_records.append({"name": player_name, "week": week, "depth_chart_pos": depth_chart_pos})






def insert_player_teams_records(relevant_players: list, start_date_mapping: dict, end_date_mapping: dict, team_id: int, season: int, team: dict, player_name_id_mapping: dict):
    """
    Functionality to generate 'player_teams' records and persist into our database 
    based on determined start and end dates of players and corresponding teams 

    Args:
        relevant_players (list): list of relevant players for given team and season 
        start_date_mapping (dict): mapping of a player name to their corresponding week they started playing for given team
        end_date_mapping (dict): mapping of a player name to their corresponding week they stopped playing for a given tema 
        team_id (int): id corresponding to players team 
        season (int): season this player_teams record corresponds to 
        team (dict): dictionary containing teams acronym & corresponding name 
        player_name_id_mapping (dict): mapping of a player name to their corresponding ID 

    """

    # generate player_teams records 
    player_teams_records = []
    for player_name in relevant_players: 
            
            # extract start & end weeks
            week_strt = start_date_mapping[player_name]
            week_end = end_date_mapping[player_name]

            player_teams_records.append({"player_id": player_name_id_mapping[player_name], "team_id": team_id, "season": season, "strt_wk": week_strt, "end_wk": week_end})
    

    filtered_player_teams_records = [record for record in player_teams_records if fetch_data.fetch_player_teams_record_by_pk(record) == None]
    if not filtered_player_teams_records:
        logging.info(f"No new player teams records to insert for the team {team['team']} in the {season} season; skipping insertion")
    else:
        insert_data.insert_player_teams_records(filtered_player_teams_records)
    


def insert_player_records(unique_player_records: list, team: dict, player_name_id_mapping: dict, season: dict):
    """
    Insert 'player' records into our databse that have not previously been persisted 

    Args:
        unique_player_records (list): unique 'player' records that can be inserted into our databse 
        team (dict): mapping of team name and acronym
        player_name_id_mapping (dict): mapping of player name and ID 
        season (dict): relevant season 
    """

    # filtered out previously inserted players
    filtered_player_records = [record for record in unique_player_records if is_previously_inserted_player(record['name'], player_name_id_mapping) == False]
    if not filtered_player_records:
        logging.info(f"No new player records to insert for the team {team['team']} in the {season} season; skipping insertion")
    else:
        insert_data.insert_players(filtered_player_records) 



def start_and_end_date_players_on_team(relevant_players: list, current_players: list, start_date_mapping: dict, end_date_mapping: dict, date_week_mapping: dict, date: str, previous_dt: str, season: int): 
    """
    Add effective dating (utilizing start and end weeks) for a players time on a given team 

    Args:
        relevant_players (list): list of relevant players for given team and season 
        current_players (list): list of players that were specified in depth chart for the given month/year 
        start_date_mapping (dict): mapping of a player name to their corresponding week they started playing for given team
        end_date_mapping (dict): mapping of a player name to their corresponding week they stopped playing for a given tema 
        date_week_mapping (dict): mapping of a particular month/date (i.e 09/01/YYYY) to a given start and end week
        date (str): date from which this archived depth chart was retrieved from
        previous_dt (str): previous date we accounted for 
        season (int): relevant year/season
    """

    # populate set if new season teams depth chart being accounted for 
    if not relevant_players: 
        relevant_players.update([player["name"] for player in current_players])
        
        # indicate start wk of these players 
        for player in relevant_players:
            start_date_mapping[player] = date_week_mapping[date]["strt_wk"]
    else: 
        curr_players = [player["name"] for player in current_players]
        
        # end date players no longer on team
        for player in list(relevant_players): 
            if player not in curr_players and player not in end_date_mapping: # ensure we don't override previously end dated players
                end_date_mapping[player] = date_week_mapping[previous_dt]["end_wk"]
        
        # start date players just added to team & add to set
        for player in curr_players:
            if player not in relevant_players: 
                start_date_mapping[player] = date_week_mapping[date]["strt_wk"]
                relevant_players.add(player)


    # end date players if its EOY, they are not already end dated, and they are in relevant players set
    if date == f"01/01/{season + 1}":
        for player in relevant_players: 
            if player not in end_date_mapping:
                end_date_mapping[player] = date_week_mapping[date]["end_wk"]




def add_unique_player_records(unique_player_records: list, curr_player_records: list):
    """
    Add unique players to list of 'unique_player_records' 

    Args:
        unique_player_records (list): the list containing previously determined unique player records
        curr_player_records (list): list of player records for the current month/year
    """

    # account for unique player records to persist 
    if not unique_player_records: 
        unique_player_records.extend(curr_player_records)
    else:
        # loop through found players and check if already present 
        for player in curr_player_records:
            if not any(existing_player["name"] == player["name"] for existing_player in unique_player_records):
                unique_player_records.append(player)



def is_previously_inserted_player(player_name: str, player_name_id_mapping: dict): 
    """
    Filtred out previously inserted players and add existing players to ID mapping if needed 

    Args:
        player_name: players name
        player_name_id_mapping (dict): mapping of names and player_ids 
    
    Returns:
        bool: whether the record exists or not 
    """
    normalized_name = player_service.normalize_name(player_name)
    player = fetch_data.fetch_player_by_normalized_name(normalized_name)
    if player == None: 
        return False
    
    # update name id mapping if needed
    if player_name not in player_name_id_mapping:
        player_name_id_mapping[player_name] = player["player_id"]
    
    return True
    



def create_date_week_mapping(season: int):
    """
    Utility function to create a mapping of relevant date to a particular start & end week 

    Args:
        season (int): the season the date should account for 
    
    Returns:
        dict: mapping of a date to a start & end week 
    """
    dates = {
        f"09/01/{season}": { "strt_wk": 1, "end_wk": 4},
        f"10/01/{season}": { "strt_wk": 5, "end_wk": 8},
        f"11/01/{season}": { "strt_wk": 9, "end_wk": 12},
        f"12/01/{season}": { "strt_wk": 13, "end_wk": 16},
        f"01/01/{season + 1}": { "strt_wk": 17, "end_wk": 18},
    }
    return dates



def extract_fantasy_relevant_players(depth_chart: BeautifulSoup): 
    """
    Retreive fantasy relevant players from a teams depth chart 

    Args:
        depth_chart (BeautifulSoup): parsed HTML containing a teams depth chart 
    
    Returns:
        list: fantasy relevant players from parsed HTML 
    """

    # loop through table rows
    relevant_positions = ["RWR", "LWR", "SWR", "WR", "TE", "QB", "RB", "IR"] 
    players = [] 

    for tr in depth_chart: 
        tds = tr.find_all("td")
        if not tds:
            continue 

        position = tds[0].get_text() 
        # skip irrelevant positions
        if position not in relevant_positions: 
            continue
        
        # update LWR/RWR to just be WR
        if position == 'LWR' or position == 'RWR' or position == 'SWR':
            position = 'WR'

        # first depth chart position will be starter
        depth_chart_pos = 1

        # loop through table cells in row & extract name & position 
        for td in tds[1:]: 
            content = td.get_text()

            # skip empty content
            if not content or content == "" or not content.strip():
                continue

            # ensure cell isn't the players #
            try: 
                int(content)
                continue # continue if successfuil conversion
            except ValueError:
                # extract position from name field if on IR 
                if position == 'IR': 
                    players_name, player_position = parse_name(content, True)
                    
                    # skip player if position isn't valid 
                    if player_position not in relevant_positions: 
                        continue

                    # in the case a players on IR, their depth chart position can indicate -1 
                    depth_chart_pos = -1
                else:
                    players_name = parse_name(content)
                        
                players.append({"name": players_name, "position": position, "depth_chart_pos": depth_chart_pos})
                depth_chart_pos += 1 # next player for current position will be further down depth chart

    return players



def parse_name(name: str, is_injured_reserve: bool = False):
    """ 
    Parses and formats a player's name. If the player is on injured reserve, also extracts their position.
    
    Args:
        name (str): Name to format, expected in "<Last>, <First> [Position]" format.
        is_injured_reserve (bool): Flag to indicate if player is on injured reserve.
    
    Returns:
        str: Formatted name if not injured reserve.
        tuple: (Formatted name, Position) if injured reserve.
    """
    parts = name.split(", ")
    first_name = parts[1].split()[0]
    last_name = parts[0]
    
    formatted_name = f"{first_name} {last_name}".title()
    formatted_name = re.sub(r"\*", "", formatted_name) # remove *'s from player name 

    if is_injured_reserve:
        position = parts[1].split()[1] if len(parts[1].split()) > 1 else None
        if position:
            position = re.sub(r"[^a-zA-Z]", "", position)  # remove non-alphabetic characters
        return formatted_name, position  

    return formatted_name  




def extract_archive_dates(archive_dates_soup: BeautifulSoup, start_year: int, end_year: int):
    """
    Helper function to extract archive dates 

    Args:
        archive_dates_soup (BeautifulSoup): parsed html containing archive dates 
        start_year (int): year to start accounting for 
        end_year (int): final year we care about
    
    Returns:
        list: mapping of season and its corresponding relevant archive dates 
    """
    logging.info(f"Extracting relevant archive dates & ids for the years {start_year} - {end_year}")
    
    archive_dates = {}

    # extract all archive dates from page
    for archive_date in archive_dates_soup: 
        
        id = archive_date.get("value")
        date_str = archive_date.get_text(strip=True)

        if id:
            date = datetime.strptime(date_str, "%m/%d/%Y").date()
            archive_dates[date] = id

    
    # create mapping of archive 
    relevant_archive_dates = [] 
    for year in range(start_year, end_year + 1): 
        yearly_archives = {"season": year}

        # relevant start/end dates for given season
        start_date = datetime.strptime(f"09/01/{year}", "%m/%d/%Y").date()
        end_date = datetime.strptime(f"01/01/{year + 1}", "%m/%d/%Y").date()

        # filter for seasonal archive dates
        filtered_archives = {
            date.strftime('%m/%d/%Y') : archive_id
            for date, archive_id in archive_dates.items() 
            if start_date <= date <= end_date
        }

        yearly_archives["archives"] = filtered_archives
        relevant_archive_dates.append(yearly_archives)

    return relevant_archive_dates