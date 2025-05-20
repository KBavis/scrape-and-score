from db import fetch_data
from datetime import datetime
import logging


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
