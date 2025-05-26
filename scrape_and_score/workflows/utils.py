from db import fetch_data, insert_data
from datetime import datetime
import logging
import random
from service import team_service
from config import props


def is_player_demographics_persisted(season: int):
    """
    Utility function to determine if player demographics records are already persisted for certain season

    Args:
        seaon (int): the season to check for 
    
    Returns:
        bool: flag indicating if necessary data still needs to be persisted
    """ 
    # player demographics
    num_pd_records = fetch_data.get_count_player_demographics_records_for_season(season)

    return num_pd_records != 0

def is_player_records_persisted(season: int):
    """
    Utility function to determine if player_teams records are already persisted for certain season

    Args:
        seaon (int): the season to check for 
    
    Returns:
        bool: flag indicating if necessary data still needs to be persisted
    """ 
    # player teams 
    num_pt_records = fetch_data.get_count_player_teams_records_for_season(season) #NOTE: If player_teams is empty, relevant players / depth_chart_position records are also assumed to be lacking for specific season since this is done in a single flow

    return num_pt_records != 0


def are_team_seasonal_metrics_persisted(season: int):
    """
    Utility function to determine if team seasonal metrics are persisted 

    Args:
        season (int): relevant season 
    """

    team_ids = [team_service.get_team_id_by_name(team['name']) for team in props.get_config('nfl.teams')]

    for id in team_ids:
        metrics = fetch_data.fetch_team_seasonal_metrics(id, season)
        
        if metrics is None:
            return False 
    
    return True


def are_player_seasonal_metrics_persisted(season: int):
    """
    Utility function to determine if player seasonal metrics are persisted 

    Args:
        season (int): relevant season 
    """

    metrics = fetch_data.fetch_player_seasonal_metrics(season)

    return True if metrics is not None else False

    


def add_stubbed_player_game_logs(player_ids: list, week: int, season: int):
    """
    Generate & insert stubbed player game logs required for predictions

    Args:
        player_ids (list): list of relevant player IDs 
        week (int): relevant week
        season (int): relevant season
    """

    logging.info(f"Attempting to insert 'player_game_log' records for Week {week} and {season} NFL Season")

    # randomly check persistence of player game log to determine if we need to persist
    player_game_log = fetch_data.fetch_player_game_log_by_pk({"player_id": random.choice(player_ids), "week": week, "year": season})
    if player_game_log is not None:
        logging.info(f"Player Game Logs corresponding to Week {week} of the {season} NFL Season already persisted; skipping insertion")
        return
    
    # fetch relevant game logs 
    game_logs = fetch_data.fetch_team_game_logs_by_week_and_season(season, week)

    # iterate through each player 
    records = []
    for player_id in player_ids:

        # fetch players team
        team_id = fetch_data.fetch_player_teams_by_week_season_and_player_id(season, week, player_id)

        # extract team game log 
        game_log = next((game_log for game_log in game_logs if game_log['team_id'] == team_id), None)
        if game_log is None:
            logging.error(f"No 'team_game_log' found corresponding to PK (team ID={team_id},week={week},season={season})")
            raise Exception("No team game log record found")
        
        records.append({"player_id": player_id, "day": game_log["day"], "week": week, "year": season, "home_team": game_log['home_team'], "opp": game_log["opp"]})
    

    # insert records 
    logging.info(f"Attempting to insert {len(records)} player_game_log records into our database")
    insert_data.insert_upcoming_player_game_logs(records) 

def generate_game_mapping(season: int, week: int): 
    """
    Generation of a mapping of a game date to the relevant teams / players

    Args:
        season (int): relevant NFL season
        week (int): relevant week during sesaon
    
    Returns:
        list: players/teams corresponding to particular game date 
    """

    team_game_logs = fetch_data.fetch_team_game_logs_by_week_and_season(season, week)
    game_logs = filter_duplicate_games(team_game_logs)

    games = []

    for game in game_logs:

        game_date = game['game_date']
        team_id = game['team_id']
        opp_id = game['opp']

        # extract fantasy relevant players corresponding to current game 
        player_ids = []
        player_ids.extend(fetch_data.fetch_players_corresponding_to_season_week_team(season, week, team_id))
        player_ids.extend(fetch_data.fetch_players_corresponding_to_season_week_team(season, week, opp_id))

        # create mapping 
        games.append({
            "game_date": game_date,
            "player_ids": player_ids,
            "team_ids": [team_id, opp_id]
        })
    
    return games


def filter_completed_games(games: list):
    """
    Util function to filter out games that have already surpassed the current date (i.e already been played)

    Args:
        games (list): list of games to filter 

    Returns:
        list: filtered games 
    """

    relevant_games = []
    curr_date = datetime.now().date()
    for game in games:

        # determine if game date has already passed 
        if curr_date > game['game_date']:
            logging.info(f"NFL Game ({game['game_date']}) for Teams {game['team_ids']} has passed; skipping daily scraping of upcoming metrics")
            continue
        
        relevant_games.append(game)
    
    return relevant_games


def filter_duplicate_games(team_game_logs: list):
    """
    Filter out team game logs that correspond to the same game

    Args:
        team_game_logs (list): list of team game logs to filter
    
    Returns:
        list: filtered game logs 
    """

    filtered_game_logs = []
    seen_team_ids = set()
    for game_log in team_game_logs: 

        team_id = game_log['team_id']        
        opp_id = game_log['opp']

        # skip game logs already accounted for 
        if team_id in seen_team_ids and opp_id in seen_team_ids:
            continue

        seen_team_ids.add(team_id)
        seen_team_ids.add(opp_id)
        filtered_game_logs.append(game_log)


    return filtered_game_logs
