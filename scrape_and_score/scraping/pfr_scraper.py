import logging
import pandas as pd
from constants import TEAM_HREFS, MONTHS, LOCATIONS, CITIES
from service import team_service, player_service, player_game_logs_service, team_game_logs_service
from config import props
from .util import fetch_page
from datetime import date, datetime
from bs4 import BeautifulSoup, Comment
from haversine import haversine, Unit
from rapidfuzz import fuzz
from db import fetch_data, insert_data


"""
Functionality to scrape relevant NFL teams and player data 

Args:
   team_and_player_data (list[dict]): every relevant fantasy NFL player corresponding to specified NFL season
   teams (list): list of unique NFL team names

Returns:
   data (tuple(list[pd.DataFrame], list[pd.DataFrame])) 
      - metrics for both players and teams
"""


def scrape_all(team_and_player_data: list, teams: list):
    # TODO (FFM-31): Create logic to determine if new player/team data avaialable. If no new team data available, skip fetching metrics and utilize persisted metrics. If no new player data available, skip fetching metrics for player.

    # fetch configs
    team_template_url = props.get_config(
        "website.pro-football-reference.urls.team-metrics"
    )
    year = props.get_config("nfl.current-year")

    # fetch relevant team metrics
    team_metrics = fetch_team_metrics(teams, team_template_url, year)

    # fetch relevant player metrics
    player_metrics = fetch_player_metrics(team_and_player_data, year)
    # return metrics
    return team_metrics, player_metrics



def scrape_historical(start_year: int, end_year: int):
    """
    Functionality to scrape player and team game logs across multiple seasons 

    Args:
        start_year (int): the starting year to scrape historical data from 
        end_year (int): the ending year to scrape historical data from 
    """
    team_template_url = props.get_config(
        "website.pro-football-reference.urls.team-metrics"
    )

    teams = team_service.get_all_teams()
    team_names = [team["name"] for team in teams]

    for year in range (start_year, end_year + 1): 
        logging.info(f"\n\nScraping team and player game logs for the {year} season")

        # # fetch team metrics for given season 
        season_team_metrics = fetch_team_metrics(team_names, team_template_url, year)
        team_game_logs_service.insert_multiple_teams_game_logs(season_team_metrics, teams, year)

        # fetch players relevant to current season 
        players = fetch_data.fetch_players_on_a_roster_in_specific_year(year)
        
        # fetch player metrics for given season
        season_player_metrics = fetch_player_metrics(players, year) 
        player_game_logs_service.insert_multiple_players_game_logs(season_player_metrics, players, year)

    
    logging.info(f"Successfully inserted all player and team game logs from {start_year} to {end_year}")



"""
Functionality to scrape the most recent team & player data based on previously persisted teams/players

Args:
    None 

Returns: 
    team_metrics, player_metrics (tuple): most recent metrics for both teams & players
"""


def scrape_recent():
    # fetch configs
    team_template_url = props.get_config(
        "website.pro-football-reference.urls.team-metrics"
    )
    year = props.get_config("nfl.current-year")

    # fetch all persisted teams
    teams = team_service.get_all_teams()
    team_names = [team["name"] for team in teams]

    # fetch recent game logs for each team
    team_metrics = fetch_team_metrics(
        team_names, team_template_url, year, recent_games=True
    )

    # fetch all persisted players
    players = player_service.get_all_players()

    # fetch recent game logs for each players
    player_metrics = fetch_player_metrics(players, year, recent_games=True)

    return team_metrics, player_metrics


"""
Functionality to fetch the metrics for each relevant player on current 53 man roster of specified year

Args:
   team_and_player_data (list[dict]) - every relevant fantasy NFL player corresponding to specified NFL season
   year (int) - year to fetch metrics for 
   recent_games (bool) - flag to determine if we are fetching metrics for most recent game or not 
"""


def fetch_player_metrics(team_and_player_data, year, recent_games=False):
    logging.info(f"Attempting to scrape player metrics for the year {year}")
    player_metrics = []
    player_urls = []

    # sort players with hashed names persisted or not to optimize URL construction
    players_with_hashed_name = [player for player in team_and_player_data if player['hashed_name'] is not None]
    players_without_hashed_name = [player for player in team_and_player_data if player['hashed_name'] is None]

    # order players by last name first initial 
    ordered_players = order_players_by_last_name(players_without_hashed_name)  # order players without a hashed name by last name first inital 

    # construct each players metrics link for players with no hashed name persisted 
    if players_without_hashed_name is not None:
        player_urls.extend(get_player_urls(ordered_players, year))

    # construct players metrics link for players with hashed name persisted 
    if players_with_hashed_name is not None:
        player_urls.extend(get_player_urls_with_hash(players_with_hashed_name, year))

    # for each player url, fetch relevant metrics
    for player_url in player_urls:
        url = player_url["url"]
        player_name = player_url["player"]
        position = player_url["position"]

        raw_html = fetch_page(url)
        if raw_html == None:  # ensure we retrieve response prior to parsing
            continue

        soup = BeautifulSoup(raw_html, "html.parser")

        # TODO (FFM-42): Gather Additional Data other than Game Logs
        logging.info(f"Fetching metrics for {position} '{player_name}'")

        game_log = get_game_log(soup, position, recent_games)
        if game_log.empty:
            logging.warn(f"Player {player_name} has no available game logs for the {year} season; skipping game logs")
            continue # skip players with no metrics 

        player_metrics.append(
            {
                "player": player_name,
                "position": position,
                "player_metrics": game_log,
            }
        )
    return player_metrics



def get_player_urls_with_hash(players: list, year: int): 
    """
    Construct player URLs when the player has a hashed previously persisted 

    Args:
        players (list): list of players with hashes persisted 
        year (int): year we want to retireve a game log for 
    
    Returns 
        list : list of player hashes 
    """

    base_url = "https://www.pro-football-reference.com/%s/%s/gamelog/%s"
    player_urls = [
        {
            "player": player['player_name'], 
            "position": player['position'], 
            "url": base_url.format(get_last_name_first_initial(player['player_name']), player['hashed_name'], year)
        } for player in players
    ]

    return player_urls


def get_last_name_first_initial(player_name: str): 
    first_and_last = player_name.split(" ")

    if len(first_and_last) < 2:
        raise Exception(f'Unable to extract first inital of last name of the players name: {player_name}')
    
    return first_and_last[1][0]
    
    

"""
Functionality to fetch the metrics for each NFL team 

Args:
   teams (list) - list of team names to fetch metrics for 
   url_template (str) - template URL used to construct specific teams URL
   year (int) - year to fetch metrics for 
   recent_games (bool) - flag to indicate if this for recent games or 

Returns:
    team_metrics (list) - list of df's containing team metrics
"""

#TODO: Rename me to fetch game logs 
def fetch_team_metrics(teams: list, url_template: str, year: int, recent_games=False):
    logging.info(f"Attempting to scrape team metrics for the following teams [{teams}]")

    team_metrics = []
    for team in teams:
        logging.info(f"Fetching metrics for the following NFL Team: '{team}'")

        # fetch raw html for team
        raw_html = get_team_metrics_html(team, year, url_template)

        # validate raw html fetched
        if raw_html == None:
            logging.error(
                f"An error occured while fetching raw HTML for the team '{team}'"
            )
            raise Exception(f"Unable to extract raw HTML for the NFL Team '{team}'")

        # get team metrics from html
        team_data = collect_team_data(team, raw_html, year, recent_games)

        # validate teams metrics were retrieved properly
        if team_data.empty:
            logging.error(
                f"An error occured while fetching metrics for the team '{team}'"
            )
            raise Exception(f"Unable to collect team data for the NFL Team '{team}'")

        # append result
        team_metrics.append({"team_name": team, "team_metrics": team_data})

    return team_metrics

#TODO: UPDATE THIS LOGIC TO ACCOUNT FOR CHANGING KEYS OVER YEARS OF DATA-STAT 
def fetch_teams_and_players_seasonal_metrics(start_year: int, end_year: int):
    teams = props.get_config("nfl.teams")
    team_template_url = props.get_config(
        "website.pro-football-reference.urls.team-metrics"
    )
    acronym_mapping = { team["name"]: team["acronym"] for team in teams }


    for year in range(start_year, end_year + 1): 
        for team in teams: 
            # fetch team ID 
            team_id = team_service.get_team_id_by_name(team["name"])

            # retrieve team page for specific year
            url = team_template_url.replace("{TEAM_ACRONYM}", acronym_mapping[team["name"]]).replace("{CURRENT_YEAR}", str(year))
            raw_html = fetch_page(url)

            # parse data 
            soup = BeautifulSoup(raw_html, "html.parser")

            # parse 'Team Stats and Rankings' table
            team_stats_table = soup.find("table", {"id": "team_stats"})
            team_stats_table_body = team_stats_table.find("tbody")
            team_stats = parse_stats(team_stats_table_body) 

            # parse 'Team Conversions' table
            team_conversions_table = soup.find("table", {"id": "team_conversions"})
            team_conversions_table_body = team_conversions_table.find("tbody")
            team_conversions = parse_conversions(team_conversions_table_body)

            # parse 'Passing' table 
            player_and_team_passing_table = soup.find("table", {"id": "passing"})
            team_passing_totals = player_and_team_passing_table.find_next("tfoot")
            player_passing_table = player_and_team_passing_table.find("tbody")
            player_passing_stats, team_passing_stats = parse_player_and_team_totals(player_passing_table, team_passing_totals)

            # parse 'Rushing and Receiving' table 
            player_and_team_rushing_and_receiving_table = soup.find("table", {"id": "rushing_and_receiving"})
            team_rushing_and_receiving_totals = player_and_team_rushing_and_receiving_table.find_next("tfoot")
            player_rushing_and_receiving_table = player_and_team_rushing_and_receiving_table.find("tbody")
            rushing_receiving_player_stats, rushing_receiving_team_stats = parse_player_and_team_totals(player_rushing_and_receiving_table, team_rushing_and_receiving_totals)

            # parse 'Kicking' table (infomration stored in a comment)
            kicking_div = soup.find('div', {'id': 'all_kicking'})
            comment = kicking_div.find(string=lambda text: isinstance(text, Comment))
            if comment:   
                table_soup = BeautifulSoup(comment, 'html.parser')
                tfoot = table_soup.find('tfoot')
                team_kicking_stats = parse_team_totals(tfoot)
            
            # parse 'Punting' table (infomration stored in a comment)
            punting_div = soup.find('div', {'id': 'all_punting'})
            comment = punting_div.find(string=lambda text: isinstance(text, Comment))
            if comment: 
                table_soup = BeautifulSoup(comment, 'html.parser')
                tfoot = table_soup.find('tfoot')
                team_punting_stats = parse_team_totals(tfoot)
            

            # parse 'Defenesne & Fumbles' table 
            defensve_div = soup.find('div', {'id': 'all_defense'})
            tfoot = defensve_div.find('tfoot')
            team_defensive_stats = parse_team_totals(tfoot)

            # parse 'Scoring Summary' 
            scoring_summary_div =  soup.find('div', {'id': 'all_scoring'})
            comment = scoring_summary_div.find(string=lambda text: isinstance(text, Comment))
            if comment: 
                table_soup = BeautifulSoup(comment, 'html.parser')
                player_tbody = table_soup.find('tbody')
                team_tfoot = table_soup.find('tfoot')
                player_scoring_summary, team_scoring_summary = parse_player_and_team_totals(player_tbody, team_tfoot)

            #TODO: Consider if we want to acocunt for Touchdown Log & Opponent Touchdown Log in future

            # insert team records 
            insert_data.format_and_insert_team_seasonal_general_metrics(team_stats, team_conversions, team_id, year)
            insert_data.insert_team_seasonal_passing_metrics(team_passing_stats, team_id, year)
            insert_data.insert_team_seasonal_rushing_and_receiving_metrics(rushing_receiving_team_stats, team_id, year)
            insert_data.insert_team_seasonal_kicking_and_punting_metrics(team_punting_stats, team_kicking_stats, team_id, year)
            insert_data.insert_team_seasonal_defense_and_fumbles_metrics(team_stats, team_defensive_stats, team_conversions, team_id, year)
            insert_data.insert_team_seasonal_scoring_metrics(team_scoring_summary, team_id, year)
            insert_data.insert_team_seasonal_rankings_metrics(team_stats, team_conversions, team_id, year)

            # generate player records 
            insert_data.insert_player_seasonal_passing_metrics(player_passing_stats, year, team_id)
            insert_data.insert_player_seasonal_rushing_and_receiving_metrics(rushing_receiving_player_stats, year, team_id)
            insert_data.insert_player_seasonal_scoring_metrics(player_scoring_summary, year, team_id)






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
   
"""
Functionality to fetch relevant metrics corresponding to a specific NFL team


All credit for the following code in this function goes to the developer of the repository:
      - https://github.com/mjk2244/pro-football-reference-web-scraper

Some subtle modifications were made to fix the repositories bug and to fit our use case.      
   
Args: 
    team (str) - NFL team full name
    raw_html (str) - raw HTML fetch for specified team
    year (int) - year to fetch metrics for 
    recent_games (bool) - flag to indicate if we are fetching most recent game or all games
      
Returns:
    pandas.DataFrame: A pandas DataFrame with relevant metrics corresponding to the specific player     
"""


def collect_team_data(team: str, raw_html: str, year: int, recent_games: bool):

    # configure data frame
    data = {
        "week": [],
        "day": [],
        "rest_days": [],
        "home_team": [],
        "distance_traveled": [],
        "opp": [],
        "result": [],
        "points_for": [],
        "points_allowed": [],
        "tot_yds": [],
        "pass_yds": [],
        "rush_yds": [],
        "opp_tot_yds": [],
        "opp_pass_yds": [],
        "opp_rush_yds": [],
    }
    df = pd.DataFrame(data)

    # create BeautifulSoup instance
    soup = BeautifulSoup(raw_html, "html.parser")

    # load game data
    games = soup.find_all("tbody")[1].find_all("tr")

    remove_uneeded_games(games, year)

    # determine how many games to process
    if recent_games:
        game_range = range(len(games) - 1, len(games))  # only last game
    else:
        game_range = range(len(games))  # all games

    # gathering data for each game
    for i in game_range:
        week = int(games[i].find("th", {"data-stat": "week_num"}).text)
        day = games[i].find("td", {"data-stat": "game_day_of_week"}).text

        rest_days = calculate_rest_days(games, i, year)

        opp = games[i].find("td", {"data-stat": "opp"}).text

        if games[i].find("td", {"data-stat": "game_location"}).text == "@":
            home_team = False
            distance_travelled = calculate_distance(
                LOCATIONS[CITIES[team]], LOCATIONS[CITIES[opp]]
            )
        else:
            home_team = True
            distance_travelled = 0

        result = games[i].find("td", {"data-stat": "game_outcome"}).text
        points_for = extract_int(games[i], "pts_off")
        points_allowed = extract_int(games[i], "pts_def")

        tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds = (
            calculate_yardage_totals(games, i)
        )

        # add row to data frame
        df.loc[len(df.index)] = [
            week,
            day,
            rest_days,
            home_team,
            distance_travelled,
            opp,
            result,
            points_for,
            points_allowed,
            tot_yds,
            pass_yds,
            rush_yds,
            opp_tot_yds,
            opp_pass_yds,
            opp_rush_yds,
        ]

    return df


"""
Helper function to calculate the yardage totals for a particular game of a team 

Args: 
    games (BeautifulSoup): parsed HTML containing game data 
    index (int): index pertaining to current game 

Returns:
    tot_yds,pass_yds,rush_yds,opp_tot_yds,opp_pass_yds,opp_rush_yds (tuple): yardage totals of particular game 
"""


def calculate_yardage_totals(games: BeautifulSoup, index: int):
    tot_yds = extract_int(games[index], "yards_off")
    pass_yds = extract_int(games[index], "pass_yds_off")
    rush_yds = extract_int(games[index], "rush_yds_off")
    opp_tot_yds = extract_int(games[index], "yards_def")
    opp_pass_yds = extract_int(games[index], "pass_yds_def")
    opp_rush_yds = extract_int(games[index], "rush_yds_def")

    return tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds


"""
Helper function to determine a teams total rest days 

Args:
    games (BeautifulSoup): parsed HTML containing game data 
    index (int): index pertaining to current game 
    year (int): year we are calculating metrics for 

Returns:
    rest_days (int): total number of rest days since previous game 

"""


def calculate_rest_days(games: BeautifulSoup, index: int, year: int):
    if index == 0:
        return 10  # set rest days to be 10 if first game of year

    # fetch previous game and current game date
    previous_game_date = (
        games[index - 1].find("td", {"data-stat": "game_date"}).text.split(" ")
    )
    current_game_date = (
        games[index].find("td", {"data-stat": "game_date"}).text.split(" ")
    )

    # account for new year
    if current_game_date[0] == "January" and previous_game_date[0] != "January":
        rest_days = date(
            year + 1, MONTHS[current_game_date[0]], int(current_game_date[1])
        ) - date(year, MONTHS[previous_game_date[0]], int(previous_game_date[1]))
    # both games in new year
    elif current_game_date[0] == "January" and previous_game_date[0] == "January":
        rest_days = date(
            year + 1, MONTHS[current_game_date[0]], int(current_game_date[1])
        ) - date(year + 1, MONTHS[previous_game_date[0]], int(previous_game_date[1]))
    # both games not in new year
    else:
        rest_days = date(
            year, MONTHS[current_game_date[0]], int(current_game_date[1])
        ) - date(year, MONTHS[previous_game_date[0]], int(previous_game_date[1]))

    return rest_days.days  # return as integer


"""
Helper function to remove all canceled/playoff games, bye weeks, 
and games yet to be played so that they aren't accounted for 

Args: 
    games (BeautifulSoup): parsed HTML containing game data 
    year (int): current eyar to account for 
"""


def remove_uneeded_games(games: BeautifulSoup, year: int):
    # remove playoff games
    j = 0
    while (
        j < len(games)
        and games[j].find("td", {"data-stat": "game_date"}).text != "Playoffs"
    ):
        j += 1
    for k in range(j, len(games)):
        games.pop()

    # remove bye weeks
    bye_weeks = []
    for j in range(len(games)):
        if games[j].find("td", {"data-stat": "opp"}).text == "Bye Week":
            bye_weeks.append(j)

    if len(bye_weeks) > 1:
        games.pop(bye_weeks[0])
        games.pop(bye_weeks[1] - 1)

    elif len(bye_weeks) == 1:
        games.pop(bye_weeks[0])

    # remove canceled games
    to_delete = []
    for j in range(len(games)):
        if games[j].find("td", {"data-stat": "boxscore_word"}).text == "canceled":
            to_delete.append(j)
    for k in to_delete:
        games.pop(k)

    # remove games that have yet to be played
    to_delete = []
    current_date = date.today()

    # skip logic if we are in the past 
    if year <= current_date.year:
        return 

    for j in range(len(games)):
        game_date = get_game_date(games[j], current_date)
        if game_date >= current_date:
            to_delete.append(j)
    for k in reversed(to_delete):  # reverse order to prevent shifting issue
        games.pop(k)


"""
Helper function to generate URL and fetch raw HTML for NFL Team

Args: 
    team_name (str) - NFL team full name
    year (int) - year to fetch raw HTML for
    url (str) - template URL to fetch HTML from
      
Returns:
    str - raw HTML from web page 
"""


def get_team_metrics_html(team_name, year, url):
    url = url.replace("{TEAM_ACRONYM}", TEAM_HREFS[team_name]).replace(
        "{CURRENT_YEAR}", str(year)
    )
    return fetch_page(url)


"""
Functionality to calculate the distance between two cities 
   
Args: 
    city1 (dict) - dictionary containing a cities latitude & longitude 
    city2 (dict) - dictionary containing a cities latitude & longitude 
      
Returns:
    double: value corresponding to the distance between the two cities  

"""


def calculate_distance(city1: dict, city2: dict):
    coordinates1 = (city1["latitude"], city1["longitude"])
    coordinates2 = (city2["latitude"], city2["longitude"])
    return haversine(coordinates1, coordinates2, unit=Unit.MILES)


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


def get_game_date(game: BeautifulSoup, current_date: date):
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


"""
Functionality to order players into a dictionary based on their last name inital

Args:
    player_data(dict): dictionary containing unique players in current season

Returns:
    ordered_players(dict) : dictionary that orders players (i.e 'A': [{<player_data>}, {<player_data>}, ...])
"""


def order_players_by_last_name(player_data: list):
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


"""
Functionality to fetch all relevant player URLs needed to extract new player metrics 

Args:
    ordered_players(dict): players ordered by their last name inital, allowing us to construct 
                            player URLs in bulk rather than one by one
    year(int): season to fetch players for 
    
Returns:
    urls(list) : list of dictionary containing players URL and name
"""


def get_player_urls(ordered_players: dict, year: int):
    base_url = "https://www.pro-football-reference.com%s/gamelog/%s"
    urls = []
    player_hashed_names = []

    for inital, player_list in ordered_players.items():
        logging.info(f"Constructing player URLs for players with the inital '{inital}'")

        # fetch players list based on last name inital
        player_list_url = "https://www.pro-football-reference.com/players/%s/" % (
            inital
        )
        soup = BeautifulSoup(
            fetch_page(player_list_url), "html.parser"
        )  # soup containing players hrefs

        # for each player in the corresponding inital, construct player URLs
        for player in player_list:
            player_name = player["player_name"]
            player_position = player["position"]

            href = get_href(
                player_name, player_position, year, soup, player_hashed_names
            )  # extract href from parsed HTML
            if href == None:
                continue
            else:
                url = base_url % (href, year)
                urls.append(
                    {"player": player_name, "position": player_position, "url": url}
                )  # append each players URL to our list of URLs

    # insert player hashed names into database 
    insert_data.update_player_hashed_name(player_hashed_names)

    return urls


"""
Functionality to fetch a specific players href, which is needed to construct their URL

Args:
    player_name (str): players name to search for 
    year (int): year corresponding to the season we are searching for metrics for 
    soup (BeautifulSoup): soup pertaining to raw HTML containing players hrefs
    player_hashed_names (list): list to add player_hashed_names records to in order to persist 

Returns:
    href (str): href needed to construct URL 
"""


def get_href(player_name: str, position: str, year: int, soup: BeautifulSoup, player_hashed_names: list):
    players = soup.find("div", id="div_players").find_all(
        "p"
    )  # find players HTML element

    for player in players:
        # Split and parse the years from player text
        years = player.text.split(" ")
        years = years[-1].split("-")  # Use the last segment for the years

        try:
            start_year, end_year = int(years[0]), int(years[1])
        except ValueError:
            logging.warning(f"Error parsing years for player {player.text}")
            continue

        # Check if the player's name, position, and year match
        player_text = player.text
        if (
            start_year <= year <= end_year
            and position in player_text
            and check_name_similarity(player_text, player_name) >= 93
        ):
            a_tag = player.find("a")
            if a_tag and a_tag.get("href"):
                href = a_tag.get("href").replace(".htm", "")
                update_players_hashed_name(player_name, href, player_hashed_names) # account for hashed name & player ID in our player_hashed_names list
                return href
            else:
                logging.warning(
                    f"Missing href for player {player_name} ({position}) in year {year}"
                )
                return None

    # TODO (FFM-40): Add Ability to Re-try Finding Players Name
    logging.warning(f"Cannot find a {position} named {player_name} from {year}")
    return None


def update_players_hashed_name(player_name: str, href: str, player_hashed_names: list): 
    """
    Helper function to extract relevant hashed name from HREF and persist for player 

    Args:
        player_name (str): player name to persist 
        href (str): href corresponding to players game logs & advanced metrics 
        player_hashed_names (list): list to update with player hashed name & ID 
    """
    # extract hashed name from href
    hashed_name_index = href.rfind('/')
    hashed_name = href[hashed_name_index + 1:]

    # extract player ID by hashed name
    normalized_name = player_service.normalize_name(player_name)
    player_id = player_service.get_player_id_by_normalized_name(normalized_name)

    if player_id is None or hashed_name is None: 
        raise Exception(f'Unable to correctly extract players hashed name or ID for player {player_name}')

    player_hashed_names.append({"hashed_name": hashed_name, "player_id": player_id})

    


"""
Helper function to determine the similarity between two names

Args:
    player_name (str): players name to compare 
    player_text (str): text from PFR containing players name

Returns:
    similarity (float): similarity of the two passed in names
"""


def check_name_similarity(player_text: str, player_name: str):
    words = player_text.split()
    name = " ".join(words[:2])
    name = name.title()
    player_name = player_name.title()
    return fuzz.partial_ratio(name, player_name)


"""
Functionality to get the game log for a player. This function 
will retrieve relevant player metrics from their game logs based 
on their position.

Args:
    soup (BeautifulSoup): parsed HTML containing relevant player metrics 
    position (str): the players corresponding position
    recent_games (bool): flag to determine if we only need to fetch game log for most recent game

Returns:
    data (pd.DataFrame): data frmae containing player game logs 
"""


def get_game_log(soup: BeautifulSoup, position: str, recent_games: bool):
    # data to retrieve for each player, regardless of position
    data = {
        "date": [],
        "week": [],
        "team": [],
        "game_location": [],
        "opp": [],
        "result": [],
        "team_pts": [],
        "opp_pts": [],
        "off_snps": [],
        "snap_pct": []
    }
    data.update(get_additional_metrics(position))  # update data with additonal metrics

    # skip players with no table
    table_body = soup.find("tbody")
    if table_body == None:
        return pd.DataFrame()

    table_rows = table_body.find_all("tr")

    # ignore inactive/DNP games
    ignore_statuses = ["Inactive", "Did Not Play", "Injured Reserve"]
    filtered_table_rows = []
    for tr in table_rows:
        elements = tr.find_all("td")

        if not elements:
            continue

        status = elements[-1].text

        # account for trade specific information
        team_cell = tr.find("td", {"data-stat": "team_name_abbr"})
        is_trade = team_cell and team_cell.has_attr("colspan") and "went from" in team_cell.text

        if status not in ignore_statuses and not is_trade:
            filtered_table_rows.append(tr)

    # check if we only want to fetch recent game log
    if recent_games:
        filtered_table_rows = filtered_table_rows[-1:]  # keep only last game log

    # add game log data to dictionary
    for tr in filtered_table_rows:
        # add common game log data
        add_common_game_log_metrics(data, tr)

        # add position specific data
        if position == "QB":
            add_qb_specific_game_log_metrics(data, tr)
        elif position == "RB":
            add_rb_specific_game_log_metrics(data, tr)
        else:
            add_wr_specific_game_log_metrics(data, tr)

    return pd.DataFrame(data=data)


"""
Functionality to retireve game log metrics for a QB

Args:
    tr (BeautifulSoup): parsed HTML tr containing player metrics 
    data (dict): dictionary containing players metrics 

Returns:
    None
"""


def add_qb_specific_game_log_metrics(data: dict, tr: BeautifulSoup):
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


"""
Functionality to retireve game log metrics for a RB

TODO (FFM-83): Account for RB Snap Percentage Metrics

Args:
    tr (BeautifulSoup): parsed HTML tr containing player metrics 
    data (dict): dictionary containing players metrics 

Returns:
    None
"""


def add_rb_specific_game_log_metrics(data: dict, tr: BeautifulSoup):

    # Add rushing and receiving stats with missing value handling
    data["rush_att"].append(extract_int(tr, "rush_att"))
    data["rush_yds"].append(extract_int(tr, "rush_yds"))
    data["rush_td"].append(extract_int(tr, "rush_td"))
    data["tgt"].append(extract_int(tr, "targets"))
    data["rec"].append(extract_int(tr, "rec"))
    data["rec_yds"].append(extract_int(tr, "rec_yds"))
    data["rec_td"].append(extract_int(tr, "rec_td"))


"""
Functionality to retrieve game log metrics for a WR

Args:
    tr (BeautifulSoup): parsed HTML tr containing player metrics 
    data (dict): dictionary containing players metrics 

Returns:
    None
"""


def add_wr_specific_game_log_metrics(data: dict, tr: BeautifulSoup):
    data["tgt"].append(extract_int(tr, "targets"))
    data["rec"].append(extract_int(tr, "rec"))
    data["rec_yds"].append(extract_int(tr, "rec_yds"))
    data["rec_td"].append(extract_int(tr, "rec_td"))


"""
Functionality to retireve common game log metrics for a given player 

Args:
    tr (BeautifulSoup): parsed HTML tr containing player metrics 
    data (dict): dictionary containing players metrics 

Returns:
    None
"""


def add_common_game_log_metrics(data: dict, tr: BeautifulSoup):
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



"""
Helper function to retrieve additional metric fields needed for a player based on position

Args:
    position (str): the players corresponding position

Returns:
    additonal_metrics (dict): additional metrics to account for based on player posiiton
"""


def get_additional_metrics(position):
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


"""
Helper function to extract int from a speciifc player metric 

Args:
    tr (BeautifulSoup): table row containing relevant metrics 

Returns:
    metric (int): derived metric converted to a int
"""


def extract_int(tr, stat):
    text = tr.find("td", {"data-stat": stat})

    if text == None:
        return 0  # return 0 if no value
    elif text.text == "":
        return 0
    else:
        return int(text.text)


"""
Helper function to extract a float from a speciifc player metric 

Args:
    tr (BeautifulSoup): table row containing relevant metrics 

Returns:
    metric (float): derived metric converted to a float 
"""


def extract_float(tr, stat):
    text = tr.find("td", {"data-stat": stat})

    if text == None:
        return 0.0
    elif text.text == "":
        return 0.0
    else:
        return float(text.text)
