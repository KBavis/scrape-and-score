import logging
from rapidfuzz import fuzz
from datetime import datetime, date
from bs4 import BeautifulSoup
from db.read.teams import fetch_team_game_log_by_pk
from haversine import haversine, Unit
from service import player_service


def add_qb_specific_game_log_metrics(data: dict, tr: BeautifulSoup):
    """
    Functionality to retireve game log metrics for a QB

    Args:
        tr (BeautifulSoup): parsed HTML tr containing player metrics 
        data (dict): dictionary containing players metrics 
    """

    data["cmp"].append(extract_int(tr, "pass_cmp"))
    data["att"].append(extract_int(tr, "pass_att"))
    data["pass_yds"].append(extract_int(tr, "pass_yds"))
    data["pass_td"].append(extract_int(tr, "pass_td"))
    data["int"].append(extract_int(tr, "pass_int"))
    data["rating"].append(extract_float(tr, "pass_rating"))
    data["sacked"].append(extract_int(tr, "pass_sacked"))
    data["rush_att"].append(extract_int(tr, "rush_att"))
    data["rush_yds"].append(extract_int(tr, "rush_yds"))
    data["rush_td"].append(extract_int(tr, "rush_td"))



def add_rb_specific_game_log_metrics(data: dict, tr: BeautifulSoup):
    """
    Functionality to retireve game log metrics for a RB

    TODO (FFM-83): Account for RB Snap Percentage Metrics

    Args:
        tr (BeautifulSoup): parsed HTML tr containing player metrics 
        data (dict): dictionary containing players metrics 
    """

    # Add rushing and receiving stats with missing value handling
    data["rush_att"].append(extract_int(tr, "rush_att"))
    data["rush_yds"].append(extract_int(tr, "rush_yds"))
    data["rush_td"].append(extract_int(tr, "rush_td"))
    data["tgt"].append(extract_int(tr, "targets"))
    data["rec"].append(extract_int(tr, "rec"))
    data["rec_yds"].append(extract_int(tr, "rec_yds"))
    data["rec_td"].append(extract_int(tr, "rec_td"))



def add_wr_specific_game_log_metrics(data: dict, tr: BeautifulSoup):
    """
    Functionality to retrieve game log metrics for a WR

    Args:
        tr (BeautifulSoup): parsed HTML tr containing player metrics 
        data (dict): dictionary containing players metrics 
    """

    data["tgt"].append(extract_int(tr, "targets"))
    data["rec"].append(extract_int(tr, "rec"))
    data["rec_yds"].append(extract_int(tr, "rec_yds"))
    data["rec_td"].append(extract_int(tr, "rec_td"))



def add_common_game_log_metrics(data: dict, tr: BeautifulSoup):
    """
    Functionality to retireve common game log metrics for a given player 

    Args:
        tr (BeautifulSoup): parsed HTML tr containing player metrics 
        data (dict): dictionary containing players metrics 

    Returns:
        None
    """
    # account for game_date element OR date element
    game_date = tr.find("td", {"data-stat": "game_date"}) 
    data["date"].append(game_date.text if game_date else tr.find("td", {"data-stat": "date"}).text)

    data["week"].append(int(tr.find("td", {"data-stat": "week_num"}).text))

    # account for team OR team_name_abbr
    team = tr.find("td", {"data-stat": "team"})
    data["team"].append(team.text if team else tr.find("td", {"data-stat": "team_name_abbr"}).text)

    data["game_location"].append(tr.find("td", {"data-stat": "game_location"}).text)

    opp = tr.find("td", {"data-stat": "opp"})
    data["opp"].append(opp.text if opp else tr.find("td", {"data-stat": "opp_name_abbr"}).text)

    # For result, team_pts, and opp_pts, split the text properly
    game_result_text = tr.find("td", {"data-stat": "game_result"}).text.split(" ")
    data["result"].append(game_result_text[0].replace(",", ""))
    data["team_pts"].append(int(game_result_text[1].split("-")[0]))
    data["opp_pts"].append(int(game_result_text[1].split("-")[1]))

    # account for # of offensive snaps and snap pct 
    snap_pct_td = tr.find("td", {"data-stat": "snap_counts_off_pct"})
    if snap_pct_td and snap_pct_td.text:
        snap_pct = float(snap_pct_td.text) 
        data["snap_pct"].append(snap_pct)
    else:
        data["snap_pct"].append(0)
    
    snap_counts_off_td = tr.find("td", {"data-stat": "snap_counts_offense"})
    if snap_counts_off_td and snap_counts_off_td.text: 
        snap_counts_off = int(snap_counts_off_td.text)
        data["off_snps"].append(snap_counts_off)
    else:
        data["off_snps"].append(0)


def construct_player_urls(players: list, season: int, base_url: str = None):
    """
    Construct relevant player URLs for specific season 

    Args:
        players (list): players to construct URLs for
        season (int): relevant season
        base_url (str, optional): base URL to utilize. Defaults to None.

    Returns:
        list: corresponding player URLs 
    """

    player_urls = []

    # sort players with hashed names persisted or not to optimize URL construction
    players_with_hashed_name = [player for player in players if player['hashed_name'] is not None]
    players_without_hashed_name = [player for player in players if player['hashed_name'] is None and player['pfr_available'] == 1] # disregard players previously indicated to be unavailable

    # log players skipped due to being unavailable
    log_disregarded_players(players)

    # check if this is for player pages or player metrics
    if base_url is not None:
        if players_without_hashed_name:
            logging.warning(f"The following players will also be skipped as there is no hashed_name persisted for them:\n\t{players_without_hashed_name}")

        return get_player_page_urls(players_with_hashed_name, base_url)

    # order players by last name first initial 
    ordered_players = order_players_by_last_name(players_without_hashed_name)  # order players without a hashed name by last name first inital 

    # construct each players metrics link for players with no hashed name persisted 
    if players_without_hashed_name is not None and len(players_without_hashed_name) > 0:
        player_urls.extend(get_player_urls(ordered_players, season))

    # construct players metrics link for players with hashed name persisted 
    if players_with_hashed_name is not None and len(players_with_hashed_name) > 0:
        player_urls.extend(get_player_urls_with_hash(players_with_hashed_name, season))
    
    return player_urls


def get_player_urls_with_hash(players: list, year: int): 
    """
    Construct player URLs when the player has a hashed previously persisted 

    Args:
        players (list): list of players with hashes persisted 
        year (int): year we want to retireve a game log for 
    
    Returns 
        list : list of player hashes 
    """

    base_url = "https://www.pro-football-reference.com/players/{}/{}/gamelog/{}"
    player_urls = [
        {
            "player": player['player_name'], 
            "position": player['position'], 
            "url": base_url.format(get_last_name_first_initial(player['player_name']), player['hashed_name'], year)
        } for player in players
    ]

    return player_urls


def get_last_name_first_initial(player_name: str): 
    """
    Get last name first inital of a players name 

    Args:
        player_name (str): relevant player name

    Returns:
        str: last name first initial 
    """

    first_and_last = player_name.split(" ")

    if len(first_and_last) < 2:
        raise Exception(f'Unable to extract first inital of last name of the players name: {player_name}')
    
    return first_and_last[1][0]


def filter_metrics_by_week(metrics: list, curr_week: int = None):
    """
    Filters out duplicate entries based on the 'week' field. Keeps the first occurrence of each week.
    
    Args:
        metrics (list): A list of dictionaries containing player metrics.
        week (int): optional arg of week 
        
    Returns:
        list: A filtered list with only the first instance for each week.
    """

    seen_weeks = set()
    filtered_metrics = []

    for metric in metrics:
        week = metric.get('week') 

        # skip irrelevant weeks if we want only a particular week
        if curr_week is not None and week != week:
            continue

        if week not in seen_weeks:
            filtered_metrics.append(metric)
            seen_weeks.add(week)

    return filtered_metrics


def get_player_page_urls(players: list, base_url: str):
    """
    Functionality to construct player page URLs 

    Args:
        players (list): the list of players to construct URLs for 
        base_url (str): the base URL to construct player page URL for 
    """

    player_urls = [
        {
            "player": player['player_name'], 
            "position": player['position'], 
            "url": base_url.format(get_last_name_first_initial(player['player_name']), player['hashed_name'])
        } for player in players
    ]

    return player_urls


def parse_team_totals(team_totals: BeautifulSoup) -> dict:
    """Parse the teams kicking and punting totals

    Args:
        team_kicking_totals (BeautifulSoup): parsed HTML containing team totals for kicking

    Returns:
        dict: key-value paris of kicking stats
    """

    team_totals_stats = {}
    prefix = "team_total_"
    tds = team_totals.find_next('tr').find_all("td")
    for td in tds:
        key = prefix + td.get('data-stat')
        value = td.get_text() 

        # skip name 
        if key == 'team_total_name_display':
            continue

        if value is not None and value != "":
            team_totals_stats[key] = value
    
    return team_totals_stats





def get_team_metrics_html(team_name, year, url):
    """
    Helper function to generate URL and fetch raw HTML for NFL Team

    Args: 
        team_name (str) - NFL team full name
        year (int) - year to fetch raw HTML for
        url (str) - template URL to fetch HTML from
        
    Returns:
        str - raw HTML from web page 
    """
    url = url.replace("{TEAM_ACRONYM}", TEAM_HREFS[team_name]).replace(
        "{CURRENT_YEAR}", str(year)
    )
    return fetch_page(url)

def parse_player_and_team_totals(players_table: BeautifulSoup, team_totals: BeautifulSoup):
    """
    Parse PFR Player and Team Totals 

    Args:
        players_table (BeautifulSoup): table containing player totals 
        team_totals (BeautifulSoup): table containing team totals 
    
    Return:
        tuple: player and team totals 
    """

    # team totals 
    team_metrics = {}
    prefix = "team_total_"
    tds = team_totals.find_next('tr').find_all("td")
    for td in tds:
        key = prefix + td.get('data-stat')
        value = td.get_text() 

        # skip name
        if key == 'team_total_name_display':
            continue

        if value is not None and value != "":
            team_metrics[key] = value

    
    # player totals
    trs = players_table.find_all("tr")
    player_metrics = {}
    for tr in trs: 
        metrics = {}

        td = tr.find('td', {'data-stat': 'name_display'})
        if td is not None:
            player_name = td.find_next('a').get_text() 
            normalized_name = player_service.normalize_name(player_name)
        else:
            # skip row if no name is present
            continue


        tds = tr.find_all("td")
        for td in tds:
            key = td.get('data-stat')

            # skip over name and position as its not a metric
            if key == 'name_display' or key == 'pos':
                continue
            value = td.get_text() 

            if value is not None and value != "":
                metrics[key] = value
        

        if metrics:
            player_metrics[normalized_name] = metrics 



    return player_metrics, team_metrics


def parse_conversions(team_conversion: BeautifulSoup):
    """
    Extract relevant team conversion ratios (i.e 3rd down, red zone, etc)

    Args:
        team_conversion (BeautifulSoup): table body containing table rows with conversion metrics 
    
    Returns:
        dict: key-values of teams conversion ratios & rankings
    """

    conversions = {} 
    trs = team_conversion.find_all("tr")
    for tr in trs: 
        header = tr.find_next("th").get_text() 

        # prefix for data stats
        if header == 'Team Stats':
            prefix = 'team_'
        elif header == 'Opp. Stats':
            prefix = 'opp_'
        elif header == 'Lg Rank Defense': 
            prefix = 'off_rank_'
        else:
            prefix = 'def_rank_'

        # loop through cells in row
        for td in tr.find_all("td"):
            key = prefix + str(td.get("data-stat")) 
            value = td.get_text() 

            if value is not None and value != '':
                conversions[key] = value


    
    return conversions


def parse_stats(team_stats_tbody: BeautifulSoup):
    """
    Functionality to extract relevant team & opponent stat totals for a given year 

    Args:
        team_stats_tbody (BeautifulSoup): parsed html 
    
    Returns:
        dict: mapping of team stats 
    """

    stats = {}
    trs = team_stats_tbody.find_all("tr")

    for tr in trs:
        header = tr.find_next("th").get_text() 

        # prefix for data stats
        if header == 'Team Stats':
            prefix = 'team_'
        elif header == 'Opp. Stats':
            prefix = 'opp_'
        elif header == 'Lg Rank Defense': 
            prefix = 'off_rank_'
        else:
            prefix = 'def_rank_'

        for td in tr: 
            stat = td.get("data-stat")

            # skip row names
            if stat == 'player':
                continue
            
            key = prefix + stat
            value = td.get_text()
            
            # ensure value exists 
            if value is not None and value != '':
                stats[key] = value
    
    return stats

def filter_update_team_game_logs(records: list):
    """
    Filter records into insertable or udpateable records 

    Args: 
        records (list): reords to filter
    
    Returns:
        tuple: (update_records, insert_records)
    """

    insert_records = []
    update_records = []

    for record in records:
        
        pk = {"team_id": record['team_id'], "week": record['week'], "year": record['season']}
        persisted_team_game_log = fetch_team_game_log_by_pk(pk)

        if not persisted_team_game_log:
            logging.info(f"No 'team_game_log' persisted corresponding to PK=[{pk}]; appending as insertable record")
            insert_records.append(record)
            continue

        # verify if changes have been made 
        if is_team_game_log_modified(record, persisted_team_game_log):
            logging.info(f"'team_game_log' correspond to PK=[{pk}] has been modified; appending as updateable record")
            update_records.append(record)
            continue
    
    return update_records, insert_records


def calculate_distance(city1: dict, city2: dict):
    """
    Functionality to calculate the distance between two cities 
    
    Args: 
        city1 (dict) - dictionary containing a cities latitude & longitude 
        city2 (dict) - dictionary containing a cities latitude & longitude 
        
    Returns:
        double: value corresponding to the distance between the two cities  

    """
    coordinates1 = (city1["latitude"], city1["longitude"])
    coordinates2 = (city2["latitude"], city2["longitude"])
    return haversine(coordinates1, coordinates2, unit=Unit.MILES)


def is_team_game_log_modified(current_game_log: dict, persisted_game_log: dict):
    """
    Verify if persisted 'team_game_log' was modfiied or not 

    Args:
        current_game_log (dict): the current gmae log 
        persisted_game_log (dict): already persisted game log 

    Returns:
        bool: flag indicated if modified or not 
    """

    keys = [
        'result', 'points_for', 'points_allowed',
        'off_tot_yds', 'off_pass_yds', 'off_rush_yds',
        'def_tot_yds', 'def_pass_yds', 'def_rush_yds',
        'pass_tds', 'pass_cmp', 'pass_att', 'pass_cmp_pct',
        'rush_att', 'rush_tds',
        'yds_gained_per_pass_att', 'adj_yds_gained_per_pass_att', 'pass_rate',
        'sacked', 'sack_yds_lost',
        'rush_yds_per_att', 'total_off_plays', 'yds_per_play',
        'fga', 'fgm', 'xpa', 'xpm',
        'total_punts', 'punt_yds',
        'pass_fds', 'rsh_fds', 'pen_fds', 'total_fds',
        'thrd_down_conv', 'thrd_down_att', 'fourth_down_conv', 'fourth_down_att',
        'penalties', 'penalty_yds',
        'fmbl_lost', 'interceptions', 'turnovers',
        'time_of_poss'
    ]

    for key in keys:
        if current_game_log[key] != persisted_game_log[key]:
            return True
        
    return False


def get_game_date(game: BeautifulSoup, current_date: date):
    """
    Functionality to fetch the game date for a game 

    Note: This application should be run on a scheduled basis throughout the football year, 
    starting in August and concluding in Feburary. Therefore, the game date year should account
    for this logic 

    Args:
        game (BeautifulSoup): BeautifulSoup object containing relevant game data 
        current_date (date) : the current date the application is being run 

    Returns:
        game_date (date) : The date corresponding to the game day    
    """

    game_date_str = game.find("td", {"data-stat": "game_date"}).text
    month_date = datetime.strptime(
        game_date_str, "%B %d"
    ).date()  # convert to date time object

    # if game date month is janurary or feburary, we must adjust logic
    if month_date.month == 1 or month_date.month == 2:
        if current_date.month == 1 or current_date.month == 2:
            game_year = (
                current_date.year
            )  # game year should be current year if application is running in january or febutary
        else:
            game_year = (
                current_date.year + 1
            )  # game yar should be next year if application is running in march through december
    else:
        game_year = (
            current_date.year
        )  # date corresponds to this year if game date isn't jan or feb
    return date(game_year, month_date.month, month_date.day)


def check_name_similarity(player_text: str, player_name: str):
    """
    Helper function to determine the similarity between two names

    Args:
        player_name (str): players name to compare 
        player_text (str): text from PFR containing players name

    Returns:
        similarity (float): similarity of the two passed in names
    """

    words = player_text.split()
    name = " ".join(words[:2])
    name = name.title()
    player_name = player_name.title()
    return fuzz.partial_ratio(name, player_name)


def order_players_by_last_name(player_data: list):
    """
    Functionality to order players into a dictionary based on their last name inital

    Args:
        player_data(dict): dictionary containing unique players in current season

    Returns:
        ordered_players(dict) : dictionary that orders players (i.e 'A': [{<player_data>}, {<player_data>}, ...])
    """

    logging.info("Ordering retrieved players by their last name's first initial")

    ordered_players = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "F": [],
        "G": [],
        "H": [],
        "I": [],
        "J": [],
        "K": [],
        "L": [],
        "M": [],
        "N": [],
        "O": [],
        "P": [],
        "Q": [],
        "R": [],
        "S": [],
        "T": [],
        "U": [],
        "V": [],
        "W": [],
        "X": [],
        "Y": [],
        "Z": [],
    }

    for player in player_data:
        last_name = player["player_name"].split()[1]
        inital = last_name[0].upper()
        ordered_players[inital].append(player)

    return ordered_players

def get_additional_metrics(position):
    """
    Helper function to retrieve additional metric fields needed for a player based on position

    Args:
        position (str): the players corresponding position

    Returns:
        additonal_metrics (dict): additional metrics to account for based on player posiiton
    """

    if position == "QB":
        additional_fields = {
            "cmp": [],
            "att": [],
            "pass_yds": [],
            "pass_td": [],
            "int": [],
            "rating": [],
            "sacked": [],
            "rush_att": [],
            "rush_yds": [],
            "rush_td": [],
        }
    elif position == "RB":
        additional_fields = {
            "rush_att": [],
            "rush_yds": [],
            "rush_td": [],
            "tgt": [],
            "rec": [],
            "rec_yds": [],
            "rec_td": [],
        }
    elif position == "WR" or position == "TE":
        additional_fields = {
            "tgt": [],
            "rec": [],
            "rec_yds": [],
            "rec_td": [],
            "snap_pct": [],
        }
    else:
        logging.error(
            f"An unexpected position was passed to get_additional_metrics: {position}"
        )
        raise Exception(
            f"The position '{position}' is not a valid position to fetch metrics for."
        )
    return additional_fields


def log_disregarded_players(players: list):
    """
    Log out players that are going to be disregarded 

    Args:
        players (list): all players active in specifid season
    """

    disregarded_players = [player['player_name'] for player in players if player['pfr_available'] == 0]
    if disregarded_players is not None and len(disregarded_players) != 0:
        logging.warning(f'The following players will be disregarded due to being unavailable in Pro-Football-Reference: \n\n{disregarded_players}\n\n')


def extract_int(tr, stat):
    """
    Helper function to extract int from a speciifc player metric 

    Args:
        tr (BeautifulSoup): table row containing relevant metrics 

    Returns:
        metric (int): derived metric converted to a int
    """
    text = tr.find("td", {"data-stat": stat})

    if text == None:
        return 0  # return 0 if no value
    elif text.text == "":
        return 0
    else:
        return int(text.text)


def extract_str(tr, stat):
    """
    Helper function to extract string from HTML 

    Args:
        tr (BeautifulSoup): soup containing stats 
        stat (str): data stat to extract 
    
    Returns:
        str (extracted string)
    """

    text = tr.find("td", {"data-stat": stat})

    if text == None:
        return ""  
    elif text.text == "":
        return ""  
    else:
        return text.text.strip()  


def extract_float(tr, stat):
    """
    Helper function to extract a float from a speciifc player metric 

    Args:
        tr (BeautifulSoup): table row containing relevant metrics 

    Returns:
        metric (float): derived metric converted to a float 
    """
    text = tr.find("td", {"data-stat": stat})

    if text == None:
        return 0.0
    elif text.text == "":
        return 0.0
    else:
        return float(text.text)


def convert_time_to_float(time_str: str) -> float:
    """
    Helper function to convert time to float 

    Args:
        time_str (str): string containing time to convert to float 
    
    Returns:
        int: converted value 
    """

    if not time_str:
        return 0.0
    
    minutes, seconds = map(int, time_str.split(":"))
    return minutes + seconds / 60