import logging
import pandas as pd
from constants import TEAM_HREFS, MONTHS, LOCATIONS, CITIES
from service import team_service, player_service, player_game_logs_service, team_game_logs_service, service_util
from config import props
from .util import fetch_page
from datetime import date, datetime
from bs4 import BeautifulSoup, Comment
from haversine import haversine, Unit
from rapidfuzz import fuzz
from datetime import datetime
from db.read.players import (
    fetch_players_on_a_roster_in_specific_year,
    fetch_player_demographic_record,
    fetch_player_date_of_birth,
    fetch_players_on_a_roster_in_specific_year_with_hashed_name,
    fetch_player_ids_of_players_who_have_advanced_metrics_persisted
)
from db.insert.players import (
    update_player_game_log_with_results,
    update_player_hashed_name,
    update_player_pfr_availablity_status,
    insert_player_dob,
    insert_player_demographics,
    insert_player_seasonal_passing_metrics,
    insert_player_seasonal_rushing_and_receiving_metrics,
    insert_player_seasonal_scoring_metrics,
    insert_player_advanced_passing_metrics,
    insert_player_advanced_rushing_receiving_metrics
)
from db.read.teams import (
    fetch_all_teams,
    fetch_team_game_log_by_pk
)
from db.insert.teams import (
    update_team_game_log_with_results,
    format_and_insert_team_seasonal_general_metrics,
    insert_team_seasonal_passing_metrics,
    insert_team_seasonal_rushing_and_receiving_metrics,
    insert_team_seasonal_kicking_and_punting_metrics,
    insert_team_seasonal_defense_and_fumbles_metrics,
    insert_team_seasonal_scoring_metrics,
    insert_team_seasonal_rankings_metrics,
)


def scrape_all(team_and_player_data: list, teams: list):
    """
    Functionality to scrape relevant NFL teams and player data 

    Args:
        team_and_player_data (list[dict]): every relevant fantasy NFL player corresponding to specified NFL season
        teams (list): list of unique NFL team names

    Returns:
        data (tuple(list[pd.DataFrame], list[pd.DataFrame])) - metrics for both players and teams
    """

    # fetch configs
    team_template_url = props.get_config(
        "website.pro-football-reference.urls.team-metrics"
    )
    year = props.get_config("nfl.current-year")

    # fetch relevant team metrics
    team_metrics = fetch_team_game_logs(teams, team_template_url, year)

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
        logging.info(f"Scraping team and player game logs for the {year} season\n\n")

        # # fetch team metrics for given season 
        season_team_metrics = fetch_team_game_logs(team_names, team_template_url, year)
        team_game_logs_service.insert_multiple_teams_game_logs(season_team_metrics, teams, year, True)

        # fetch players relevant to current season 
        players = fetch_players_on_a_roster_in_specific_year(year)
        
        # fetch player metrics for given season
        season_player_metrics = fetch_player_metrics(players, year) 
        player_game_logs_service.insert_multiple_players_game_logs(season_player_metrics, players, year)

    
    logging.info(f"Successfully inserted all player and team game logs from {start_year} to {end_year}")


def update_game_logs_and_insert_advanced_metrics(week: int, season: int):
    """
    Update 'player_game_log' and 'team_game_log' records with relevant results and insert 
    relevant advanced player metrics into our database 


    Args:
        week (int): relevant week to account for 
        season (int): relevant season to account for 
    """

    # scrape & persist team game logs 
    update_team_game_logs(week, season)

    # scprae & persist player game logs
    update_player_game_logs(week, season)

    # scprae & persist player advanced metrics 
    scrape_player_advanced_metrics(season, season, week)


def update_player_game_logs(week: int, season: int):
    """"
    Functionality to update player game logs with relevant outcomes OR remove unecessary stubbed player game logs 

    TODO (FFM-311): Remove Stubbed Player Game Log if No Metrics Retrieved in Results Workflow 

    Args: 
        week (int): relevant week 
        season (int): relevant season 
    """

    logging.info(f"Attempting to update Week {week} of the {season} NFL Season 'player_game_log' records with the results ")
    player_metrics = []
    
    players = fetch_players_on_a_roster_in_specific_year(season)
    player_urls = construct_player_urls(players, season)

    # for each player url, fetch relevant metrics
    for player_url in player_urls:
        url = player_url["url"]
        player_name = player_url["player"]
        position = player_url["position"]

        raw_html = fetch_page(url)
        if raw_html == None:  
            continue

        soup = BeautifulSoup(raw_html, "html.parser")

        # retrieve game logs
        game_log = get_game_log(soup, position)
        if game_log.empty:
            logging.warning(f"Player {player_name} has no available game logs for the {season} season; skipping updating player game logs")
            continue 

        # filter out non-relevant weeks 
        filtered_df = game_log[game_log['week'] == week]
        if filtered_df.empty:
            logging.info(f"Player {player_name} has no 'player_game_log' available corresponding to week {week} of the {season} NFL season; skipping updating game log with results")
            continue

        # validate length of filtered df 
        if filtered_df.shape[0] > 1:
            raise Exception(f'Found multiple game logs corresponding to week {week} of the {season} NFL season for player {player_name}')
        
        game = filtered_df.iloc[0]

        # create updateable record 
        player_metrics.append(
            {
                "player_id": player_service.get_player_id_by_normalized_name(player_service.normalize_name(player_name)),
                "week": week,  
                "year": season,
                "result": game.get("result", None),
                "points_for": game.get("points_for", None),
                "points_allowed": game.get("points_allowed", None),
                "completions": game.get("completions", None),
                "attempts": game.get("attempts", None),
                "pass_yd": game.get("pass_yd", None),
                "pass_td": game.get("pass_td", None),
                "interceptions": game.get("interceptions", None),
                "rating": game.get("rating", None),
                "sacked": game.get("sacked", None),
                "rush_att": game.get("rush_att", None),
                "rush_yds": game.get("rush_yds", None),
                "rush_tds": game.get("rush_tds", None),
                "tgt": game.get("tgt", None),
                "rec": game.get("rec", None),
                "rec_yd": game.get("rec_yd", None),
                "rec_td": game.get("rec_td", None),
                "snap_pct": game.get("snap_pct", None),
                "fantasy_points": game.get("fantasy_points", None),
                "off_snps": game.get("off_snps", None),
            }
        )

    if player_metrics:
        logging.info(f"Attempting to update {len(player_metrics)} player_game_log records with results")
        update_player_game_log_with_results(player_metrics)

    
def update_team_game_logs(week: int, season: int):
    """
    Scrape & persist team game log results 

    Args:
        week (int): relevant week
        season (int): relevant season
    """

    teams = fetch_all_teams()
    team_names = [team['name'] for team in teams]
    
    base_url = props.get_config("website.pro-football-reference.urls.team-metrics")

    records = []

    # iterate through each team
    for team in team_names:
        logging.info(f"Updating 'team_game_log' with results for [Team={team}, Week={week}, Season={season}]")

        html = get_team_metrics_html(team, season, base_url)
        if html is None:
            logging.error(f"Unable to retrieve HTML for Team '{team}' for Week {week} of the {season} NFL Season")
            raise Exception('Unable to retrieve HTML for Team Game Log')
    
        soup = BeautifulSoup(html, "html.parser")
        updates = extract_team_game_log_updates(soup, week, season)

        # skip accounting for updates if none found 
        if updates is None:
            continue

        # update record with team ID 
        updates['team_id'] = team_service.get_team_id_by_name(team)

        print(f"Team game logs record scraped for Week {week} of {season} NFL Season:\n\t{updates}")

        # append record
        records.append(updates)
    

    update_records, insert_records = filter_update_team_game_logs(records)
    if update_records:
        logging.info(f"Updating {len(update_records)} 'team_game_log' records with results")
        update_team_game_log_with_results(update_records)

    if insert_records:
        raise Exception(f'Found unique game logs when attempting to update team game logs with results: {insert_records}')


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


def extract_team_game_log_updates(soup: BeautifulSoup, week: int, season: int):
    """
    Extract relevant metrics pertaining to a particular NFL team for a given week/season

    Args:
        soup (BeautifulSoup): parsed HTMl
        week (int): relevant week
        season (int): relevant season

    Returns:
        dict: team game log corresponding to team/week/season
    """

    games = soup.find_all("tbody")
    if games and len(games) > 1:
        games = games[1].find_all("tr")
    else:
        logging.info(f"No team game log records found for week {week} and season {season}")
        return None

    if not games or len(games) < 1:
        logging.warning(f"No games found for week {week} and season {season}; skipping extraction of results")
        return None
    
    
    game_range = range(len(games)) 
    for i in game_range:

        week_element = games[i].find("th", {"data-stat": "week_num"})
        curr_week = int(week_element.text) if week_element else extract_int(games[i], 'week_num')
        if not curr_week or curr_week != week:
            continue

        # extract normal team/opp metrics 
        result = extract_str(games[i], "game_outcome") 
        points_for = extract_int(games[i], "pts_off")
        points_allowed = extract_int(games[i], "pts_def")
        off_tot_yds = extract_int(games[i], "yards_off")
        off_pass_yds = extract_int(games[i], "pass_yds_off")
        off_rush_yds = extract_int(games[i], "rush_yds_off")
        def_tot_yds = extract_int(games[i], "yards_def")
        def_pass_yds = extract_int(games[i], "pass_yds_def")
        def_rush_yds = extract_int(games[i], "rush_yds_def")
        
        # extract advanced team metrics 
        #TODO (FFM-312): Account for pro-football-reference no longer supplying these metrics 
        (
            pass_tds, pass_cmp, pass_att, pass_cmp_pct,
            rush_att, rush_tds, yds_gained_per_pass_att,
            adj_yds_gained_per_pass_att, pass_rate, sacked,
            sack_yds_lost, rush_yds_per_att, total_off_plays,
            yds_per_play, fga, fgm, xpa, xpm,
            total_punts, punt_yds, pass_fds, rsh_fds, pen_fds,
            total_fds, thrd_down_conv, thrd_down_att, fourth_down_conv,
            fourth_down_att, penalties, penalty_yds, fmbl_lost,
            interceptions, turnovers, time_of_poss
        ) = extract_advanced_metrics(games, i)


        team_game_log = {
            "week": week,
            "season": season,
            "result": result,
            "points_for": points_for,
            "points_allowed": points_allowed,
            "off_tot_yds": off_tot_yds,
            "off_pass_yds": off_pass_yds,
            "off_rush_yds": off_rush_yds,
            "def_tot_yds": def_tot_yds,
            "def_pass_yds": def_pass_yds,
            "def_rush_yds": def_rush_yds,
            "pass_tds": pass_tds,
            "pass_cmp": pass_cmp,
            "pass_att": pass_att,
            "pass_cmp_pct": pass_cmp_pct,
            "rush_att": rush_att,
            "rush_tds": rush_tds,
            "yds_gained_per_pass_att": yds_gained_per_pass_att,
            "adj_yds_gained_per_pass_att": adj_yds_gained_per_pass_att,
            "pass_rate": pass_rate,
            "sacked": sacked,
            "sack_yds_lost": sack_yds_lost,
            "rush_yds_per_att": rush_yds_per_att,
            "total_off_plays": total_off_plays,
            "yds_per_play": yds_per_play,
            "fga": fga,
            "fgm": fgm,
            "xpa": xpa,
            "xpm": xpm,
            "total_punts": total_punts,
            "punt_yds": punt_yds,
            "pass_fds": pass_fds,
            "rsh_fds": rsh_fds,
            "pen_fds": pen_fds,
            "total_fds": total_fds,
            "thrd_down_conv": thrd_down_conv,
            "thrd_down_att": thrd_down_att,
            "fourth_down_conv": fourth_down_conv,
            "fourth_down_att": fourth_down_att,
            "penalties": penalties,
            "penalty_yds": penalty_yds,
            "fmbl_lost": fmbl_lost,
            "interceptions": interceptions,
            "turnovers": turnovers,
            "time_of_poss": time_of_poss
        }

        return team_game_log


    
def scrape_and_persist_player_demographics(season: int): 
    """
    Functionality to scrape & persist 'player_demographic' records for players in an upcoming season

    NOTE: This functionality should be executed after new players have been accounted for 

    Args:
        season (int): the season to scrape & persist for 
    """

    logging.info(f"Scraping & persisted player demographics for the {season} NFL Season")

    # scrape & persist each teams new draftees hashed_names & check if in PFR 
    scrape_and_persist_team_draftees_hashed_names(season)

    base_url = props.get_config('website.pro-football-reference.urls.player-page')

    # fetch relevant players
    players = fetch_players_on_a_roster_in_specific_year(season)

    # construct URLs 
    player_urls = construct_player_urls(players, season, base_url)

    for url in player_urls:

        html = fetch_page(url['url'])
        if html is None: 
            logging.warning(f"No valid HTML retrieved for player '{url['player']}'")
            continue

        soup = BeautifulSoup(html, "html.parser")

        parse_and_insert_player_demographics_and_dob(soup, url['player'], season)


def scrape_and_persist_team_draftees_hashed_names(season: int):
    """
    Functionality to scrape & persist the hashed_names in PFR for new team draftees 

    Args:
        season (int): relevant season 
    """

    # generate URLs 
    base_url = props.get_config('website.pro-football-reference.urls.team-draft')
    teams = props.get_config('nfl.teams')

    urls = [ base_url.format(team['acronym'], season) for team in teams ]
    
    players_to_update = []
    relevant_positions = ['QB', 'RB', 'TE', 'WR']

    for url in urls:

        # scrape team page containing draftees 
        html = fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")

        # parse HTML element containing draftee information
        draftees_table = soup.find("table", {'id': 'draft'})
        draftees = draftees_table.find_next("tbody")
        draft_records = draftees.find_all('tr')

        # iterate through each player drafteed
        for player in draft_records:

            # skip players that aren't fantasy relevant
            position = player.find('td', {'data-stat': 'pos'}).text
            if position not in relevant_positions:
                continue
            
            # extract HREF from HTML
            player_element = player.find('td', {'data-stat': 'player'})
            player_link = player_element.find_next('a')

            player_name = player_link.text 
            player_href = player_link.get('href')

            # parse out relevant hashed name & append to records to update
            hashed_name = player_href.split("/")[-1].replace(".htm", "")

            # fetch player ID corresponding to player 
            normalized_name = player_service.normalize_name(player_name)
            player_id = player_service.get_player_id_by_normalized_name(normalized_name)

            if player_id is None:
                # retry with additional details
                player_id = player_service.get_player_id_by_position_season_and_normalized_name(season, position, normalized_name)
                if player_id is None:
                    logging.warning(f'Unable to extract the player_id for player: {player_name}; skipping update of player hashed_name')
                    continue

            players_to_update.append({"hashed_name": hashed_name, "player_id": player_id})

    # update db with hashed names
    if players_to_update:
        update_player_hashed_name(players_to_update)
        player_ids = [id['player_id'] for id in players_to_update]
        update_player_pfr_availablity_status(player_ids, True)



def fetch_player_metrics(team_and_player_data, year, recent_games=False):
    """
    Functionality to fetch the metrics for each relevant player on current 53 man roster of specified year

    Args:
        team_and_player_data (list[dict]) - every relevant fantasy NFL player corresponding to specified NFL season
        year (int) - year to fetch metrics for 
        recent_games (bool) - flag to determine if we are fetching metrics for most recent game or not 
    """

    logging.info(f"Attempting to scrape player metrics for the year {year}")
    player_metrics = []

    player_urls = construct_player_urls(team_and_player_data, year)

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
            logging.warning(f"Player {player_name} has no available game logs for the {year} season; skipping game logs")
            continue # skip players with no metrics 

        # logic to account for player demographics 
        parse_and_insert_player_demographics_and_dob(soup, player_url['player'], year)

        player_metrics.append(
            {
                "player": player_name,
                "position": position,
                "player_metrics": game_log,
            }
        )
    return player_metrics


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


def parse_and_insert_player_demographics_and_dob(soup: BeautifulSoup, player_name: str, year: int):
    """
    Check if player demographics record from pro-football-reference are persisted, if not, persist

    Args:
        soup (BeautifulSoup): parsed HTML of players page
        player_name (str): players name that we want to check/insert record for 
        year (int): relevant season
    """

    logging.info(f"Checking if player '{player_name}' in the {year} season has a pro-football-reference player_demographics record inserted")

    # retreive player ID by players normalized name 
    normalized_name = player_service.normalize_name(player_name)
    player_id = player_service.get_player_id_by_normalized_name(normalized_name)

    # retrieve player demographics record
    player_demographics_record = fetch_player_demographic_record(player_id, year)

    if not player_demographics_record:
        logging.info(f'No previously inserted player_demographic record exists for player {player_name} for the {year} NFL season; parsing & inserting record.')

        player_dob = fetch_player_date_of_birth(player_id)

        # parse & insert players date of birth if not already persisted
        if player_dob is None:
            logging.info('No player date of birth persisted; updating player record with players date of birth')
            player_meta_data_div = soup.find('div', {'id': 'info'})
            player_dob_el = player_meta_data_div.find(id='necro-birth')

            # verify relevant information is on page
            if player_dob_el is None:
                logging.warning(f"No DOB Element Exists for player {player_name}; skipping insertion of player demographics")
                return
            
            player_dob = player_dob_el.get("data-birth")
            insert_player_dob(player_id, player_dob)
        
        # calculate age of player for when season starts (9/1/{year})
        season_start = datetime.strptime(f"{year}-09-01", '%Y-%m-%d').date()
        players_birth = datetime.strptime(player_dob, '%Y-%m-%d').date()
        age = int((season_start - players_birth).days / 365)

        # extract height & weight from HTML 
        meta_data_div = soup.find('div', {'id': 'meta'})
        paragraph_elements = meta_data_div.find_all('p')

        height_weight_p = paragraph_elements[2]
        height_weight_spans = height_weight_p.find_all('span')

        height = height_weight_spans[0].text
        weight = height_weight_spans[1].text

        ft_and_inches = height.split('-')
        player_height = int(ft_and_inches[0]) * 12 + int(ft_and_inches[1])

        player_weight = int(weight.replace('lb', ''))

        logging.info(f'Attempting to insert player demographics record for player {player_name} for the {year} season --> Age: {age}, Height: {height}, Weight: {weight}')
        insert_player_demographics(player_id, year, age, player_height, player_weight)

    else: # skip insertion if record already exists
        logging.info(f"Player_demographics record exists for player '{player_name}' in the {year} NFL season; skipping insertion")
        return

def log_disregarded_players(players: list):
    """
    Log out players that are going to be disregarded 

    Args:
        players (list): all players active in specifid season
    """

    disregarded_players = [player['player_name'] for player in players if player['pfr_available'] == 0]
    if disregarded_players is not None and len(disregarded_players) != 0:
        logging.warning(f'The following players will be disregarded due to being unavailable in Pro-Football-Reference: \n\n{disregarded_players}\n\n')


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
    
    

def fetch_team_game_logs(teams: list, url_template: str, year: int, recent_games=False):
    """
    Functionality to fetch team game logs for each NFL team 

    Args:
        teams (list) - list of team names to fetch metrics for 
        url_template (str) - template URL used to construct specific teams URL
        year (int) - year to fetch metrics for 
        recent_games (bool) - flag to indicate if this for recent games or 

    Returns:
        team_metrics (list) - list of df's containing team metrics
    """
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
            logging.warning(
                f"No team data was retreived for Team '{team}'"
            )
            continue

        # append result
        team_metrics.append({"team_name": team, "team_metrics": team_data})

    return team_metrics

def fetch_teams_and_players_seasonal_metrics(start_year: int, end_year: int):
    """
    Fetch teams and player seasonal metrics

    Args:
        start_year (int): year to start scraping metrics from
        end_year (int): year to stop scraping metrics for 
    """

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

            # insert team records 
            format_and_insert_team_seasonal_general_metrics(team_stats, team_conversions, team_id, year)
            insert_team_seasonal_passing_metrics(team_passing_stats, team_id, year)
            insert_team_seasonal_rushing_and_receiving_metrics(rushing_receiving_team_stats, team_id, year)
            insert_team_seasonal_kicking_and_punting_metrics(team_punting_stats, team_kicking_stats, team_id, year)
            insert_team_seasonal_defense_and_fumbles_metrics(team_stats, team_defensive_stats, team_conversions, team_id, year)
            insert_team_seasonal_scoring_metrics(team_scoring_summary, team_id, year)
            insert_team_seasonal_rankings_metrics(team_stats, team_conversions, team_id, year)

            # generate player records 
            insert_player_seasonal_passing_metrics(player_passing_stats, year, team_id)
            insert_player_seasonal_rushing_and_receiving_metrics(rushing_receiving_player_stats, year, team_id)
            insert_player_seasonal_scoring_metrics(player_scoring_summary, year, team_id)


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


def scrape_player_advanced_metrics(start_year: int, end_year: int, week: int = None): 
    """
    Extract player player advanced metrics across multiple years

    Args:
        start_year (int): year to start scraping metrics for
        end_year (int): year to stop scraping metrics
        week (int, optional): relevant week to account for. Defaults to None.
    """

    for year in range(start_year, end_year + 1):
        logging.info(f"Scraping player advanced passing, rushing, and receiving metrics for the {year} season")
        
        # NOTE: All players that we want to extract advanced metrics for SHOULD have already had their game logs persisted, thus, the hashed names should be present (O/W, we can skip)
        players = fetch_players_on_a_roster_in_specific_year_with_hashed_name(year)

        # filter previously inserted records 
        players_already_persisted = fetch_player_ids_of_players_who_have_advanced_metrics_persisted(year, week)
        players = [player for player in players if player['player_id'] not in players_already_persisted ]
        logging.info(f"Players to fetch metrics for filtered down to length {len(players)} for season {year} of the NFL season")



        # generate URLs 
        base_url = props.get_config('website.pro-football-reference.urls.advanced-metrics')
        player_urls = [
            {"player_id": player['player_id'], "player_name": player['player_name'], "url": base_url.format(get_last_name_first_initial(player['player_name']), player['hashed_name'], year)}     
            for player in players
        ]

        # scrape and persist advanced passing, rushing, and receiving metrics 
        for player_url in player_urls:
            html = fetch_page(player_url["url"])
            soup = BeautifulSoup(html, "html.parser")

            # advanced_passing table 
            advanced_passing_table = soup.find("table", {"id": "passing_advanced"})

            # resilience for table id name change
            if advanced_passing_table is None:
                advanced_passing_table = soup.find("table", {"id": "adv_passing"})
                
            if advanced_passing_table is not None: 
                logging.info(f"Scraping advanced passing metrics for player {player_url['player_name']} for the {year} season")
                
                # parse stats for player and persist 
                advanced_passing_metrics = parse_advanced_passing_table(advanced_passing_table)
                if advanced_passing_metrics is None:
                    logging.warn(f"No passing metrics for player {player_url['player_name']} and year {year} were not retreived; skipping insertion\n\n")
                    continue
                
                filtered_metrics = filter_metrics_by_week(advanced_passing_metrics, week)
                insert_player_advanced_passing_metrics(filtered_metrics, player_url['player_id'], year)

            # rushing & receiving table 
            advanced_rushing_receiving_table = soup.find("table", {"id": "adv_rushing_and_receiving"})
            
            # resilince for table id name change 
            if advanced_rushing_receiving_table is None:
                advanced_rushing_receiving_table = soup.find("table", {"id": "adv_rushing_and_receiving"})

            if advanced_rushing_receiving_table is not None: 
                logging.info(f"Scraping advanced rushing/receiving metrics for player {player_url['player_name']} for the {year} season")

                # parse rushing/receiving 
                advanced_rushing_receiving_metrics = parse_advanced_rushing_receiving_table(advanced_rushing_receiving_table)
                if advanced_rushing_receiving_metrics is None:
                    logging.warn(f"No rushing/receiving metrics for player {player_url['player_name']} and season {year} were not retreived; skipping insertion\n\n")
                    continue
                filtered_metrics = filter_metrics_by_week(advanced_rushing_receiving_metrics, week)
                insert_player_advanced_rushing_receiving_metrics(filtered_metrics, player_url['player_id'], year)


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


def parse_advanced_passing_table(table: BeautifulSoup):
    """
    Parse relevant advanced passing metrics for a player in a given season.

    Args:
        table (BeautifulSoup): Advanced passing table.

    Returns:
        list: List of dictionaries containing passing metrics for the entire season,
              or None if no valid data is found.
    """

    table_body = table.find_next("tbody")
    if table_body is None:
        logging.warning("Table body for advanced passing table is null")
        return None
    
    table_rows = table_body.find_all("tr")
    if not table_rows:
        logging.warning("Table rows for advanced passing table are null or empty")
        return None
    
    metrics = []
    required_stats = {
        "week": "week_num", "age": "age", "first_downs": "pass_first_down",
        "first_down_passing_per_pass_play": "pass_first_down_pct",
        "intended_air_yards": "pass_target_yds",
        "intended_air_yards_per_pass_attempt": "pass_tgt_yds_per_att",
        "completed_air_yards": "pass_air_yds",
        "completed_air_yards_per_cmp": "pass_air_yds_per_cmp",
        "completed_air_yards_per_att": "pass_air_yds_per_att",
        "yds_after_catch": "pass_yac",
        "yds_after_catch_per_cmp": "pass_yac_per_cmp",
        "drops": "pass_drops", "drop_pct": "pass_drop_pct",
        "poor_throws": "pass_poor_throws",
        "poor_throws_pct": "pass_poor_throw_pct",
        "sacked": "pass_sacked", "blitzed": "pass_blitzed",
        "hurried": "pass_hurried", "hits": "pass_hits",
        "pressured": "pass_pressured", "pressured_pct": "pass_pressured_pct",
        "scrmbl": "rush_scrambles",
        "yds_per_scrmbl": "rush_scrambles_yds_per_att"
    }
    
    # loop through entirety of table rows 
    for row in table_rows:
        table_data = row.find_all("td")

        # skip row if no table data present
        if not table_data:
            continue
        
        weekly_metrics = {}

        found_any = False  # track if at least one attribute was found

        for key, stat in required_stats.items():
            stat_element = row.find("td", {"data-stat": stat})
            if stat_element:
                metric = stat_element.get_text(strip=True)

                # ensure metric isn't empty
                if metric == '':
                    metric = 0

                weekly_metrics[key] = metric
                if key != 'week':
                    found_any = True  # at least one attribute was found other than week
            else:
                # resilience for week data-stat name change
                if key == 'week': 
                    stat_element = row.find("td", {"data-stat": "team_game_num_season"})
                    if stat_element:
                        weekly_metrics[key] = stat_element.get_text(strip=True)
        
        # append only if at least one attribute was successfully retrieved
        if found_any:
            metrics.append(weekly_metrics)
        else:
            logging.warning("Skipping row as all advanced passing attributes were missing.")

    return metrics if metrics else None



def parse_advanced_rushing_receiving_table(table: BeautifulSoup):
    """
    Parse relevant advanced rushing and receiving metrics for a player in a given season

    args:
        table (BeautifulSoup): advanced rushing and receiving table
    """

    table_body = table.find_next("tbody")
    if table_body is None:
        logging.warning("table body for advanced rushing and receiving table is null")
        return None

    table_rows = table_body.find_all("tr")
    if not table_rows:
        logging.warning("table rows for advanced rushing and receiving table are null or empty")
        return None

    metrics = []
    required_stats = {
        "week": "week_num", "age": "age", "rush_first_downs": "rush_first_down",
        "rush_yds_before_contact": "rush_yds_before_contact",
        "rush_yds_before_contact_per_att": "rush_yds_bc_per_rush",
        "rush_yds_after_contact": "rush_yac",
        "rush_yds_after_contact_per_att": "rush_yac_per_rush",
        "rush_brkn_tackles": "rush_broken_tackles",
        "rush_att_per_brkn_tackle": "rush_broken_tackles_per_rush",
        "rec_first_downs": "rec_first_down", "yds_before_catch": "rec_air_yds",
        "yds_before_catch_per_rec": "rec_air_yds_per_rec",
        "yds_after_catch": "rec_yac", "yds_after_catch_per_rec": "rec_yac_per_rec",
        "avg_depth_of_tgt": "rec_adot", "rec_brkn_tackles": "rec_broken_tackles",
        "rec_per_brkn_tackle": "rec_broken_tackles_per_rec",
        "dropped_passes": "rec_drops", "drop_pct": "rec_drop_pct",
        "int_when_tgted": "rec_target_int", "qbr_when_tgted": "rec_pass_rating"
    }

    for row in table_rows:
        table_data = row.find_all("td")

        # skip row if no data is present
        if not table_data:
            continue

        weekly_metrics = {}
        found_any = False  # track if at least one attribute was found

        for key, stat in required_stats.items():
            stat_element = row.find("td", {"data-stat": stat})
            if stat_element:
                metric = stat_element.get_text(strip=True)

                # ensure metric isn't empty
                if metric == '':
                    metric = 0

                weekly_metrics[key] = metric
                if key != 'week':
                    found_any = True  # at least one attribute was found other than week
            else:
                # resilience for week data-stat name change
                if key == 'week': 
                    stat_element = row.find("td", {"data-stat": "team_game_num_season"})
                    if stat_element:
                        weekly_metrics[key] = stat_element.get_text(strip=True)
        

        # append only if at least one attribute was found
        if found_any:
            metrics.append(weekly_metrics)
        else:
            logging.warning("Skipping row as all advanced rushing/receiving attributes were missing")

    return metrics if metrics else None



def collect_team_data(team: str, raw_html: str, year: int, recent_games: bool):
    """
    Functionality to fetch relevant metrics corresponding to a specific NFL team

    Args: 
        team (str) - NFL team full name
        raw_html (str) - raw HTML fetch for specified team
        year (int) - year to fetch metrics for 
        recent_games (bool) - flag to indicate if we are fetching most recent game or all games
        
    Returns:
        pandas.DataFrame: A pandas DataFrame with relevant metrics corresponding to the specific player     
    """

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
        "pass_tds": [],
        "pass_cmp": [],
        "pass_att": [],
        "pass_cmp_pct": [],
        "rush_att": [],
        "rush_tds": [],
        "yds_gained_per_pass_att": [],
        "adj_yds_gained_per_pass_att": [],
        "pass_rate": [],
        "sacked": [],
        "sack_yds_lost": [],
        "rush_yds_per_att": [],
        "total_off_plays": [],
        "yds_per_play": [],
        "fga": [],
        "fgm": [],
        "xpa": [],
        "xpm": [],
        "total_punts": [],
        "punt_yds": [],
        "pass_fds": [],
        "rsh_fds": [],
        "pen_fds": [],
        "total_fds": [],
        "thrd_down_conv": [],
        "thrd_down_att": [],
        "fourth_down_conv": [],
        "fourth_down_att": [],
        "penalties": [],
        "penalty_yds": [],
        "fmbl_lost": [],
        "interceptions": [],
        "turnovers": [],
        "time_of_poss": []
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
        week_element = games[i].find("th", {"data-stat": "week_num"})
        if not week_element:
            week = extract_int(games[i], 'week_num')
        else:
            week = int(week_element.text)

        day = extract_str(games[i], 'game_day_of_week')

        rest_days = calculate_rest_days(games, i, year)

        opp_element = games[i].find("td", {"data-stat": "opp"})
        if not opp_element:
            opp_element = games[i].find("td", {"data-stat": "opp_name_abbr"})
            opp_text = opp_element.find_next('a').text
            opp = service_util.get_team_name_by_pfr_acronym(opp_text)
        else:
            opp = opp_element.text

        if games[i].find("td", {"data-stat": "game_location"}).text == "@":
            home_team = False
            distance_travelled = calculate_distance(
                LOCATIONS[CITIES[team]], LOCATIONS[CITIES[opp]]
            )
        else:
            home_team = True
            distance_travelled = 0

        result_element = games[i].find("td", {"data-stat": "game_outcome"})
        if result_element:
            result = result_element.text
        else:
            result_element = games[i].find("td", {"data-stat": "team_game_result"})
            result = result_element.text



        points_for = extract_int(games[i], "points")
        points_allowed = extract_int(games[i], "points_opp")

        tot_yds, pass_yds, rush_yds = (
            calculate_yardage_totals(games, i)
        )


        # extract advanced metrics 
        (
            pass_tds, pass_cmp, pass_att, pass_cmp_pct,
            rush_att, rush_tds, yds_gained_per_pass_att,
            adj_yds_gained_per_pass_att, pass_rate, sacked,
            sack_yds_lost, rush_yds_per_att, total_off_plays,
            yds_per_play, fga, fgm, xpa, xpm,
            total_punts, punt_yds, pass_fds, rsh_fds, pen_fds,
            total_fds, thrd_down_conv, thrd_down_att, fourth_down_conv,
            fourth_down_att, penalties, penalty_yds, fmbl_lost,
            interceptions, turnovers, time_of_poss
        ) = extract_advanced_metrics(games, i)


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
            pass_tds,
            pass_cmp,
            pass_att,
            pass_cmp_pct,
            rush_att,
            rush_tds,
            yds_gained_per_pass_att,
            adj_yds_gained_per_pass_att,
            pass_rate,
            sacked,
            sack_yds_lost,
            rush_yds_per_att,
            total_off_plays,
            yds_per_play,
            fga,
            fgm,
            xpa,
            xpm,
            total_punts,
            punt_yds,
            pass_fds,
            rsh_fds,
            pen_fds,
            total_fds,
            thrd_down_conv,
            thrd_down_att,
            fourth_down_conv,
            fourth_down_att,
            penalties,
            penalty_yds,
            fmbl_lost,
            interceptions,
            turnovers,
            time_of_poss
        ]

    return df


def extract_advanced_metrics(games: BeautifulSoup, index: int):
    """
    Helper functon to extract 'advanced' metrics (i.e metrics we originally didn't consider :P)

    Args:
        games (BeautifulSoup): parsed HTML containing game data 
        index (int): index pertaining to current game 
    
    Returns: 
        tuple: all advanced metrics 
    """

    pass_tds = extract_int(games[index], "pass_td")
    pass_cmp = extract_int(games[index], "pass_cmp")
    pass_att = extract_int(games[index], "pass_att")
    pass_cmp_pct = extract_float(games[index], "pass_cmp_pct")
    rush_att = extract_int(games[index], "rush_att")
    rush_tds = extract_int(games[index], "rush_td")
    yds_gained_per_pass_att = extract_float(games[index], "pass_yds_per_att")
    adj_yds_gained_per_pass_att = extract_float(games[index], "pass_adj_yds_per_att")
    pass_rate = extract_float(games[index], "pass_rating")
    sacked = extract_int(games[index], "pass_sacked")
    sack_yds_lost = extract_int(games[index], "pass_sacked_yds")
    rush_yds_per_att = extract_float(games[index], "rush_yds_per_att")
    total_off_plays = extract_int(games[index], "plays_offense")
    yds_per_play = extract_float(games[index], "yds_per_play_offense")
    fga = extract_int(games[index], "fga")
    fgm = extract_int(games[index], "fgm")
    xpa = extract_int(games[index], "xpa")
    xpm = extract_int(games[index], "xpm")
    total_punts = extract_int(games[index], "punt")
    punt_yds = extract_int(games[index], "punt_yds")
    pass_fds = extract_int(games[index], "first_down_pass")
    rsh_fds = extract_int(games[index], "first_down_rush")
    pen_fds = extract_int(games[index], "first_down_penalty")
    total_fds = extract_int(games[index], "first_down")
    thrd_down_conv = extract_int(games[index], "third_down_success")
    thrd_down_att = extract_int(games[index], "third_down_att")
    fourth_down_conv = extract_int(games[index], "fourth_down_success")
    fourth_down_att = extract_int(games[index], "fourth_down_att")
    penalties = extract_int(games[index], "penalties")
    penalty_yds = extract_int(games[index], "penalties_yds")
    fmbl_lost = extract_int(games[index], "fumbles_lost")
    int = extract_int(games[index], "pass_int")
    turnovers = extract_int(games[index], "turnovers")
    time_of_poss = convert_time_to_float(extract_str(games[index], "time_of_poss"))

    return (
        pass_tds, pass_cmp, pass_att, pass_cmp_pct, rush_att, rush_tds,
        yds_gained_per_pass_att, adj_yds_gained_per_pass_att, pass_rate, sacked,
        sack_yds_lost, rush_yds_per_att, total_off_plays, yds_per_play,
        fga, fgm, xpa, xpm, total_punts, punt_yds, pass_fds, rsh_fds, pen_fds, total_fds,
        thrd_down_conv, thrd_down_att, fourth_down_conv, fourth_down_att, penalties,
        penalty_yds, fmbl_lost, int, turnovers, time_of_poss
    )



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


def calculate_yardage_totals(games: BeautifulSoup, index: int):
    """
    Helper function to calculate the yardage totals for a particular game of a team 

    Args: 
        games (BeautifulSoup): parsed HTML containing game data 
        index (int): index pertaining to current game 

    Returns:
        tot_yds,pass_yds,rush_yds (tuple): yardage totals of particular game 
    """
    tot_yds = extract_int(games[index], "tot_yds")
    pass_yds = extract_int(games[index], "pass_yds")
    rush_yds = extract_int(games[index], "rush_yds")

    return tot_yds, pass_yds, rush_yds


def calculate_rest_days(games: list, index: int, year: int):
    """
    Helper function to determine a teams total rest days 

    Args:
        games (BeautifulSoup): parsed HTML containing game data 
        index (int): index pertaining to current game 
        year (int): year we are calculating metrics for 

    Returns:
        rest_days (int): total number of rest days since previous game 
    """

    if index == 0:
        return 10  # set rest days to be 10 if first game of year

    previous_game_str = games[index - 1].find("td", {"data-stat": "date"}).a.text
    current_game_str = games[index].find("td", {"data-stat": "date"}).a.text

    previous_game_date = datetime.strptime(previous_game_str, "%Y-%m-%d").date()
    current_game_date = datetime.strptime(current_game_str, "%Y-%m-%d").date()

    rest_days = (current_game_date - previous_game_date).days

    return rest_days


def remove_uneeded_games(games: BeautifulSoup, year: int):
    """
    Helper function to remove all canceled/playoff games, bye weeks, 
    and games yet to be played so that they aren't accounted for 

    Args: 
        games (BeautifulSoup): parsed HTML containing game data 
        year (int): current eyar to account for 
    """
    # remove playoff games
    j = 0
    while j < len(games):
        game_date_td = games[j].find("td", {"data-stat": "game_date"})

        if game_date_td is None:
            j += 1
            continue

        if game_date_td.text == "Playoffs":
            break

        j += 1

    # Only remove games from the Playoffs onward (if Playoffs was found)
    games[j:] = []


    # remove bye weeks
    bye_weeks = []
    for j in range(len(games)):
        opp_td = games[j].find("td", {"data-stat": "opp"})
        if opp_td is not None and opp_td.text == "Bye Week":
            bye_weeks.append(j)

    if len(bye_weeks) > 1:
        games.pop(bye_weeks[0])
        games.pop(bye_weeks[1] - 1)
    elif len(bye_weeks) == 1:
        games.pop(bye_weeks[0])

    # remove canceled games
    to_delete = []
    for j in range(len(games)):
        boxscore_td = games[j].find("td", {"data-stat": "boxscore_word"})
        if boxscore_td is not None and boxscore_td.text == "canceled":
            to_delete.append(j)

    # Reverse delete to avoid index shifting
    for k in reversed(to_delete):
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



def get_player_urls(ordered_players: dict, year: int):
    """
    Functionality to fetch all relevant player URLs needed to extract new player metrics 

    Args:
        ordered_players(dict): players ordered by their last name inital, allowing us to construct 
                                player URLs in bulk rather than one by one
        year(int): season to fetch players for 
        
    Returns:
        urls(list) : list of dictionary containing players URL and name
    """

    base_url = "https://www.pro-football-reference.com%s/gamelog/%s"
    urls = []
    player_hashed_names = []
    pfr_unavailable_player_ids = []

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
                player_name, player_position, year, soup, player_hashed_names, pfr_unavailable_player_ids
            )  # extract href from parsed HTML
            if href == None:
                continue
            else:
                url = base_url % (href, year)
                urls.append(
                    {"player": player_name, "position": player_position, "url": url}
                )  # append each players URL to our list of URLs

    # insert player hashed names into database 
    update_player_hashed_name(player_hashed_names)

    # update players who are not available in pfr for future optimziations 
    if pfr_unavailable_player_ids is not None:
        update_player_pfr_availablity_status(pfr_unavailable_player_ids)

    return urls



def get_href(player_name: str, position: str, year: int, soup: BeautifulSoup, player_hashed_names: list, pfr_unavailable_player_ids: list):
    """
    Functionality to fetch a specific players href, which is needed to construct their URL

    Args:
        player_name (str): players name to search for 
        year (int): year corresponding to the season we are searching for metrics for 
        soup (BeautifulSoup): soup pertaining to raw HTML containing players hrefs
        player_hashed_names (list): list to add player_hashed_names records to in order to persist 
        pfr_unavailable_player_ids (list): list of player IDs that correspond to players unavailable in PFR

    Returns:
        href (str): href needed to construct URL 
    """

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

    # account for player being unavailable 
    pfr_unavailable_player_ids.append(player_service.get_player_id_by_normalized_name(player_service.normalize_name(player_name)))
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




def get_game_log(soup: BeautifulSoup, position: str, recent_games: bool):
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
