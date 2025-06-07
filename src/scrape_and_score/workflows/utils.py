from datetime import datetime
import logging
import random
from service import team_service
from config import props
import os
from db.read.players import (
    get_count_player_demographics_records_for_season,
    get_count_player_teams_records_for_season,
    fetch_player_seasonal_metrics,
    fetch_player_game_log_by_pk,
    fetch_player_teams_by_week_season_and_player_id,
    fetch_players_corresponding_to_season_week_team,
)
from db.read.teams import (
    fetch_team_seasonal_metrics,
    fetch_team_game_logs_by_week_and_season,
)
from db.insert.players import insert_upcoming_player_game_logs


def is_player_demographics_persisted(season: int):
    """
    Utility function to determine if player demographics records are already persisted for certain season

    Args:
        seaon (int): the season to check for

    Returns:
        bool: flag indicating if necessary data still needs to be persisted
    """

    # player demographics
    num_pd_records = get_count_player_demographics_records_for_season(season)

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
    num_pt_records = get_count_player_teams_records_for_season(
        season
    )  # NOTE: If player_teams is empty, relevant players / depth_chart_position records are also assumed to be lacking for specific season since this is done in a single flow

    return num_pt_records != 0


def are_team_seasonal_metrics_persisted(season: int):
    """
    Utility function to determine if team seasonal metrics are persisted

    Args:
        season (int): relevant season
    """

    team_ids = [
        team_service.get_team_id_by_name(team["name"])
        for team in props.get_config("nfl.teams")
    ]

    for id in team_ids:
        metrics = fetch_team_seasonal_metrics(id, season)

        if metrics is None:
            return False

    return True


def are_player_seasonal_metrics_persisted(season: int):
    """
    Utility function to determine if player seasonal metrics are persisted

    Args:
        season (int): relevant season
    """

    metrics = fetch_player_seasonal_metrics(season)

    return True if metrics is not None else False


def add_stubbed_player_game_logs(player_ids: list, week: int, season: int):
    """
    Generate & insert stubbed player game logs required for predictions

    Args:
        player_ids (list): list of relevant player IDs
        week (int): relevant week
        season (int): relevant season
    """

    logging.info(
        f"Attempting to insert 'player_game_log' records for Week {week} and {season} NFL Season"
    )

    # randomly check persistence of player game log to determine if we need to persist
    player_game_log = fetch_player_game_log_by_pk(
        {"player_id": random.choice(player_ids), "week": week, "year": season}
    )
    if player_game_log is not None:
        logging.info(
            f"Player Game Logs corresponding to Week {week} of the {season} NFL Season already persisted; skipping insertion"
        )
        return

    # fetch relevant game logs
    game_logs = fetch_team_game_logs_by_week_and_season(season, week)

    # iterate through each player
    records = []
    for player_id in player_ids:

        # fetch players team
        team_id = fetch_player_teams_by_week_season_and_player_id(
            season, week, player_id
        )

        # extract team game log
        game_log = next(
            (game_log for game_log in game_logs if game_log["team_id"] == team_id), None
        )
        if game_log is None:
            logging.error(
                f"No 'team_game_log' found corresponding to PK (team ID={team_id},week={week},season={season})"
            )
            raise Exception("No team game log record found")

        records.append(
            {
                "player_id": player_id,
                "day": game_log["day"],
                "week": week,
                "year": season,
                "home_team": game_log["home_team"],
                "opp": game_log["opp"],
            }
        )

    # insert records
    logging.info(
        f"Attempting to insert {len(records)} player_game_log records into our database"
    )
    insert_upcoming_player_game_logs(records)


def generate_game_mapping(season: int, week: int):
    """
    Generation of a mapping of a game date to the relevant teams / players

    Args:
        season (int): relevant NFL season
        week (int): relevant week during sesaon

    Returns:
        list: players/teams corresponding to particular game date
    """

    team_game_logs = fetch_team_game_logs_by_week_and_season(season, week)
    game_logs = filter_duplicate_games(team_game_logs)

    games = []

    for game in game_logs:

        game_date = game["game_date"]
        team_id = game["team_id"]
        opp_id = game["opp"]

        # extract fantasy relevant players corresponding to current game
        player_ids = []
        player_ids.extend(
            fetch_players_corresponding_to_season_week_team(season, week, team_id)
        )
        player_ids.extend(
            fetch_players_corresponding_to_season_week_team(season, week, opp_id)
        )

        # create mapping
        games.append(
            {
                "game_date": game_date,
                "player_ids": player_ids,
                "team_ids": [team_id, opp_id],
            }
        )

    return games


def get_position_features():
    """
    Retrieve relevant features utilized in the training of our position specific neural network model

    Args:
        None

    Returns:
        tuple: (qb_features, rb_features, wr_features, te_features): each relevant features specific features that are being utilized to make predictions
    """

    logging.info(
        "Attempting to extract relevant positional features utilized in the training of relevant Neural Networks"
    )

    dir = "data/inputs"

    # validate relevant directory exists
    if not os.path.exists(dir):
        raise Exception(f"Please ensure that the directory, '{dir}', exists.")

    # validate necessary files are present
    all_files = os.listdir(dir)
    if not all_files:
        raise Exception(
            f"Please ensure that you first train your neural network models by passing in the --train argument when invoking the application prior to generating predictions"
        )

    # extract file names
    files = [f for f in all_files if os.path.isfile(os.path.join(dir, f))]

    # validate that each position has features file
    positions = {"QB", "RB", "WR", "TE"}
    feature_files = {}
    now = datetime.now()

    for file in files:

        parts = file.split("_")
        if len(parts) < 4:
            continue  # skip irrelevant files

        position = parts[0]
        if position not in positions:
            continue

        # extract datetime and parse
        try:
            date_str = parts[2] + parts[3].split(".")[0]
            file_dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
        except ValueError:
            continue

        # keep file name with datetime closest to current time
        if position in feature_files:
            prev_dt, _ = feature_files[position]
            if abs(now - file_dt) < abs(now - prev_dt):
                feature_files[position] = (file_dt, file)
        else:
            feature_files[position] = (file_dt, file)

    # validate all positions have features
    missing_pos = positions - feature_files.keys()
    if missing_pos:
        raise Exception(
            f"Unable to retrieve features for positions: {missing_pos}. "
            f"Please invoke this application with the --train flag to generate relevant feature files."
        )

    # extract relevant features from each files

    position_features = {
        position: extract_file_contents(feature_files[position][1], dir)
        for position in positions
    }
    return position_features


def extract_file_contents(file: str, dir: str):
    """
    Helper function to transform contents of our feautres txt files into a Python list

    Args:
        file (str): relevant file name to extract features from
        dir (str): relevant directory where file resides

    Returns:
        list: list of features
    """

    with open(os.path.join(dir, file), "r") as f:
        lines = [line.strip() for line in f]

    return lines


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
        if curr_date > game["game_date"]:
            logging.info(
                f"NFL Game ({game['game_date']}) for Teams {game['team_ids']} has passed; skipping daily scraping of upcoming metrics"
            )
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

        team_id = game_log["team_id"]
        opp_id = game_log["opp"]

        # skip game logs already accounted for
        if team_id in seen_team_ids and opp_id in seen_team_ids:
            continue

        seen_team_ids.add(team_id)
        seen_team_ids.add(opp_id)
        filtered_game_logs.append(game_log)

    return filtered_game_logs
