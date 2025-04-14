from config import props

"""
Module to act as intermediary between business logic & data access layer for player_game_logs 

TODO: Rename this file to something else since its transforming into a larger scope than just player_game_logs
"""
import logging
from db import insert_data, fetch_data
import pandas as pd
from . import team_service, service_util

"""
Functionality to insert multiple players game logs 

Args:
   player_metrics (list): list of dictionaries containing player's name, position, and game_logs
   depth_charts (list): list of dictionaries containing a plyer's name & corresponding player_id

Returns:
   None
"""


def insert_multiple_players_game_logs(player_metrics: list, depth_charts: list, year: int = None):

    remove_previously_inserted_games(player_metrics, depth_charts, year)

    if len(player_metrics) == 0:
        logging.info("No new player game logs to persist; skipping insertion")
        return

    for player in player_metrics:
        player_name = player["player"]
        logging.info(f"Attempting to insert game logs for player '{player_name}'")

        # extract position
        position = player["position"]

        # ensure metrics recorded
        df = player["player_metrics"]
        if df.empty:
            logging.warning(
                f"No player metrics corresponding to player '{player_name}; skipping insertion"
            )
            continue

        # fetch players ID
        player_id = get_player_id_by_name(player_name, depth_charts)

        # iterate through game logs, generate tuples, and insert into db
        if position == "QB":
            tuples = get_qb_game_log_tuples(df, player_id, year)
            insert_data.insert_qb_player_game_logs(tuples)
        elif position == "RB":
            tuples = get_rb_game_log_tuples(df, player_id, year)
            insert_data.insert_rb_player_game_logs(tuples)
        elif position == "TE" or position == "WR":
            tuples = get_wr_or_te_game_log_tuples(df, player_id, year)
            insert_data.insert_wr_or_te_player_game_logs(tuples)
        else:
            raise Exception(
                f"Unknown position '{position}'; unable to fetch game log tuples"
            )


"""
Utility function to check if a player game logs table is empty

Args:
   None

Returns:
   bool: truthy value to determine if table is empty or not
   
"""


def is_player_game_logs_empty():
    player_game_log = fetch_data.fetch_one_player_game_log()

    if player_game_log == None:
        return True
    else:
        return False


# TODO (FFM-84): Refactor this code so that utility functions in seperate file

"""
Utility function to retrieve a players ID based on their name 

Args:
   depth_chart (list): list of dictionary items to iterate through and retrieve id for 
   player_name (str): the player name to retrieve ID for 
   
Returns:
   player_id (int): id corresponding to the specified player
"""


def get_player_id_by_name(player_name: str, depth_charts: list):
    player = next(
        (player for player in depth_charts if player["player_name"] == player_name),
        None,
    )
    return player["player_id"] if player else None


"""
Utility function to fetch game log tuples to insert into our database for a QB

Args: 
   df (pd.DataFrame): data frame to extract into tuples
   player_id (int): player ID to acount for 
   year (int): season the game log pertains to 

Returns:
   tuples (list): list of tuples to be directly inserted into our database
"""


def get_qb_game_log_tuples(df: pd.DataFrame, player_id: int, year):
    tuples = []
    for _, row in df.iterrows():

        game_log = (
            player_id,
            row["week"],
            service_util.extract_day_from_date(row["date"]),
            year,
            service_util.extract_home_team_from_game_location(row["game_location"]),
            team_service.get_team_id_by_name(
                service_util.get_team_name_by_pfr_acronym(row["opp"])
            ),
            row["result"],
            row["team_pts"],
            row["opp_pts"],
            row["cmp"],
            row["att"],
            row["pass_yds"],
            row["pass_td"],
            row["int"],
            row["rating"],
            row["sacked"],
            row["rush_att"],
            row["rush_yds"],
            row["rush_td"],
            row["snap_pct"],
            row["off_snps"]
        )
        tuples.append(game_log)

    return tuples


"""
Utility function to fetch game log tuples to insert into our database for a RB

Args: 
   df (pd.DataFrame): data frame to extract into tuples
   player_id (int): player ID to acount for 
   year (int): season the game log pertains to 
   team_name_mapping (dict): look up table for team names that have changed over years


Returns:
   tuples (list): list of tuples to be directly inserted into our database
"""


def get_rb_game_log_tuples(df: pd.DataFrame, player_id: int, year: int):
    tuples = []
    for _, row in df.iterrows():

        game_log = (
            player_id,
            row["week"],
            service_util.extract_day_from_date(row["date"]),
            year,
            service_util.extract_home_team_from_game_location(row["game_location"]),
            team_service.get_team_id_by_name(
                service_util.get_team_name_by_pfr_acronym(row["opp"])
            ),
            row["result"],
            row["team_pts"],
            row["opp_pts"],
            row["rush_att"],
            row["rush_yds"],
            row["rush_td"],
            row["tgt"],
            row["rec"],
            row["rec_yds"],
            row["rec_td"],
            row["snap_pct"],
            row["off_snps"]
        )
        tuples.append(game_log)

    return tuples


"""
Utility function to fetch game log tuples to insert into our database for a WR or TE

Args: 
   df (pd.DataFrame): data frame to extract into tuples
   player_id (int): player ID to acount for 
   year (int): season the game log pertains to 

Returns:
   tuples (list): list of tuples to be directly inserted into our database
"""


def get_wr_or_te_game_log_tuples(df: pd.DataFrame, player_id: int, year: int):
    tuples = []
    for _, row in df.iterrows():

        game_log = (
            player_id,
            row["week"],
            service_util.extract_day_from_date(row["date"]),
            year,
            service_util.extract_home_team_from_game_location(row["game_location"]),
            team_service.get_team_id_by_name(
                service_util.get_team_name_by_pfr_acronym(row["opp"])
            ),
            row["result"],
            row["team_pts"],
            row["opp_pts"],
            row["tgt"],
            row["rec"],
            row["rec_yds"],
            row["rec_td"],
            row["snap_pct"],
            row["off_snps"]
        )
        tuples.append(game_log)

    return tuples


"""
Functionality to determine if a game log was persisted for a given week 

Args:
   game_log_pk (dict): PK to check is persisted in DB 

Returns:
   game_log (dict): None or persisted game log
"""


def is_game_log_persisted(game_log_pk: dict):
    game_log = fetch_data.fetch_player_game_log_by_pk(game_log_pk)

    if game_log == None:
        return False
    else:
        return True


"""
Utility function to remove games that have already been persisted

Args:
   player_metrics (list): list of dictionaries containing player's name, position, and game_logs
   depth_charts (list): list of dictionaries containing a plyer's name & corresponding player_id
   year (int): season corresponding to game logs

Returns:
   None
"""


def remove_previously_inserted_games(player_metrics, depth_charts, year):
    player_metric_pks = []

    # generate pks for each team game log
    for player in player_metrics:
        df = player["player_metrics"]
        if len(df) == 1:
            player_metric_pks.append(
                {
                    "player_id": get_player_id_by_name(player["player"], depth_charts),
                    "week": str(df.iloc[0]["week"]),
                    "year": year,
                }
            )

    # check if this execution is for recent games or not
    if len(player_metrics) != len(player_metric_pks):
        logging.info(
            "Program execution is not for most recent games; skipping check for previously persisted player game logs"
        )
        return

    # remove duplicate entires
    index = 0
    while index < len(player_metrics):
        if is_game_log_persisted(player_metric_pks[index]):
            del player_metrics[index]
            del player_metric_pks[index]
        else:
            logging.debug(
                f"Player game log corresponding to PK [{player_metric_pks[index]}] not persisted; inserting new game log"
            )
            index += 1


"""
Functionaltiy to calculate the fantasy points for players 

TODO (FFM-128): Account for 2-Pt Conversions

Args:
   recent_game (bool): flag to indicate if we should only be accounting for most recent game or all games 
   start_year (int): either the start year of a range, or the sole year to account for 
   end_year (int): the end year of the range to calculate fantasy points for 

Returns:
   None
"""


def calculate_fantasy_points(recent_game: bool, start_year: int, end_year: int = None):
    # determine if this is for recent week, single year, or a range of years
    if recent_game:
        logging.info(f"Recent game flag passed; calculating fantasy points for most recent week in the year {start_year}")
        player_game_logs = fetch_data.fetch_all_player_game_logs_for_recent_week(start_year)
    else:
        if end_year == None:
            logging.info(f"Calculating fantasy points for the year {start_year}")
            player_game_logs = fetch_data.fetch_all_player_game_logs_for_given_year(start_year)
        else:
            logging.info(f"Calculating fantasy points from the year {start_year} to the year {end_year}")
            player_game_logs = []
            for curr_year in range(start_year, end_year + 1):
                curr_year_game_logs = fetch_data.fetch_all_player_game_logs_for_given_year(curr_year)
                player_game_logs.extend(curr_year_game_logs)
    
    logging.info(f"Successfully fetched {len(player_game_logs)} player game logs")

    (
        passing_yd_pts,
        passing_td_pts,
        passing_int_pts,
        rushing_yd_pts,
        rushing_td_pts,
        receiving_yd_pts,
        receiving_td_pts,
        receiving_pts,
    ) = get_offensive_point_configs()

    players_points = []
    for player_game_log in player_game_logs:
        points = calculate_point_total(
            player_game_log,
            passing_yd_pts,
            passing_td_pts,
            passing_int_pts,
            rushing_yd_pts,
            rushing_td_pts,
            receiving_yd_pts,
            receiving_td_pts,
            receiving_pts,
        )
        players_points.append(
            {
                "player_id": player_game_log["player_id"],
                "week": player_game_log["week"],
                "year": player_game_log["year"],
                "fantasy_points": points,
            }
        )

    insert_fantasy_points(players_points)


"""
Functionality to calculate the total points a player


TODO (FFM-128): Account for 2-Pt Conversions and Fumbles 

Args:
   player_game_log (dict): item containing all relevant stats needed to make calculations 
   passing_yd_pts (float): # of fantasy points for a passing yard
   passing_td_pts (int): # of fantasy points for a passing td 
   passing_int_pts (int): # of fantasy points for a passing interception
   rushing_yd_pts (float): # of fantasy points for a rushing yard
   rushing_td_pts (int): # of fantasy points for a rushing td 
   receiving_yd_pts (float): # of fantasy points for a receiving yard
   receiving_td_pts (int): # of fantasy points for a receiving td 
   receiving_pts (int): # of fantasy points for a reception
   
Returns:
   points (float): total fantasy points for a player based on their game log
"""


def calculate_point_total(
    player_game_log,
    passing_yd_pts,
    passing_td_pts,
    passing_int_pts,
    rushing_yd_pts,
    rushing_td_pts,
    receiving_yd_pts,
    receiving_td_pts,
    receiving_pts,
):
    return round(
        ((player_game_log.get("interceptions", 0) or 0) * passing_int_pts)
        + ((player_game_log.get("pass_yd", 0) or 0) * passing_yd_pts)
        + ((player_game_log.get("pass_td", 0) or 0) * passing_td_pts)
        + ((player_game_log.get("rush_yds", 0) or 0) * rushing_yd_pts)
        + ((player_game_log.get("rush_tds", 0) or 0) * rushing_td_pts)
        + ((player_game_log.get("rec_yd", 0) or 0) * receiving_yd_pts)
        + ((player_game_log.get("rec_td", 0) or 0) * receiving_td_pts)
        + ((player_game_log.get("rec", 0) or 0) * receiving_pts),
        2,
    )


"""
Functionality to insert a fantasy points into our DB 

Args:
   points (list): list of dictionaries containing player_game_log PK & corresponding fantasy points for that week

Returns:
   None
"""


def insert_fantasy_points(points: list):
    if points == []:
        logging.info("No fantasy points were calculated; skipping insertion")
        return

    logging.info(f"Attempting to insert players calculated fantasy points into our DB")
    insert_data.add_fantasy_points(points)


"""
Functionality to calculate and persist the weekly averages for players. The average calculated 
will only take into account the current season AND the games that have been played in this current season 
based on the current week 

TODO: Deprecated function with due to the creation of new update_player_weekly_agg DB function

Args:
    start_week (int): week to start calculating averages for 
    end_week (int): week to stop calcualting averages for 
    season (int): season to calculate averages for 

Returns:
    None 
"""
def calculate_weekly_fantasy_point_averages(start_week: int, end_week: int, season: int): 
    # retrieve active players in specified season 
    players = fetch_data.fetch_players_active_in_specified_year(season)

    player_agg_metrics = []

    # calculate weekly averages (i.e fantasy points, advanced metrics, etc)
    for player in players: 
        player_id = player['id']

        # fetch aggregate fantasy points for each week
        for curr_week in range(start_week, end_week + 1): 
            logging.info(f"Calculating weekly fantasy point averages from week 1 to week {curr_week} in the {season} season for player {player['name']}.")


            # weekly fantasy point 
            weekly_fantasy_points = fetch_data.fetch_player_fantasy_points(player_id, season, curr_week)
            if not any(record["week"] == curr_week for record in weekly_fantasy_points):
                logging.info(f"Skipping week {curr_week} for player {player['name']} as no data is available.")
                continue
            
            weeks_active = len(weekly_fantasy_points)   
            total_fpts = sum(record["fantasy_points"] for record in weekly_fantasy_points)

            avg_fpts = round(float(total_fpts) / weeks_active, 2) if weeks_active > 0 else 0.0

            player_agg_metrics.append({"player_id": player_id, "week": curr_week, "season": season, "fantasy_points": avg_fpts})
    
    
    # insert aggregate fantasy point avgs per week for specified season 
    logging.info(f"Attempting to insert aggregate fantasy points for the {season} NFL season.")
    insert_data.insert_player_weekly_aggregate_metrics(player_agg_metrics)



"""
Utility function to retrieve fantasy points scoring configs 

Args:
   None

Returns:
   passing_yd_pts, passing_td_pts, passing_int_pts, rushing_yd_pts, rushing_td_pts, receiving_yd_pts, receiving_td_pts, receiving_pts (tuple): 
      scoring system for offensive player
"""


def get_offensive_point_configs():
    passing_yd_pts = props.get_config("scoring.offense.passing.yard")
    passing_td_pts = props.get_config("scoring.offense.passing.td")
    passing_int_pts = props.get_config("scoring.offense.passing.int")

    rushing_yd_pts = props.get_config("scoring.offense.rushing.yard")
    rushing_td_pts = props.get_config("scoring.offense.rushing.td")

    receiving_yd_pts = props.get_config("scoring.offense.receiving.yard")
    receiving_td_pts = props.get_config("scoring.offense.receiving.td")
    receiving_pts = props.get_config("scoring.offense.receiving.rec")

    logging.info(
        f"Configured offensive fantasy points scoring: "
        f"[Passing Yd: {passing_yd_pts}, Passing TD: {passing_td_pts}, "
        f"Passing INT: {passing_int_pts}, Rushing Yd: {rushing_yd_pts}, "
        f"Rushing TD: {rushing_td_pts}, Receiving Yd: {receiving_yd_pts}, "
        f"Receiving TD: {receiving_td_pts}, Receiving Rec: {receiving_pts}]"
    )

    return (
        passing_yd_pts,
        passing_td_pts,
        passing_int_pts,
        rushing_yd_pts,
        rushing_td_pts,
        receiving_yd_pts,
        receiving_td_pts,
        receiving_pts,
    )
