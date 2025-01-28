import logging
import pandas as pd
from . import service_util, team_service
from db import insert_data, fetch_data
from config import props

"""
Functionality to insert multiple teams game logs 

Args: 
   team_metrics (list): list of dictionaries containing team's name & game logs 
   teams_and_ids (list): list of dictionaries containing team's name & corresponding team ID 
   
Returns:
   None
"""


def insert_multiple_teams_game_logs(team_metrics: list, teams_and_ids: list):

    curr_year = props.get_config("nfl.current-year")  # fetch current year

    remove_previously_inserted_game_logs(team_metrics, curr_year, teams_and_ids)

    if len(team_metrics) == 0:
        logging.info("No new team game logs to persist; skipping insertion")
        return

    for team in team_metrics:
        team_name = team["team_name"]
        logging.info(f"Attempting to insert game logs for team '{team_name}")

        df = team["team_metrics"]
        if df.empty:
            logging.warning(
                f"No team metrics corresponding to team '{team_name}; skipping insertion"
            )
            continue

        # fetch team ID
        team_id = get_team_id_by_name(team_name, teams_and_ids)

        # fetch team game logs
        tuples = get_team_log_tuples(df, team_id, curr_year)

        # insert team game logs into db
        insert_data.insert_team_game_logs(tuples)


"""
Utility function to fetch team game log tuples to insert into our database 

Args: 
   df (pd.DataFrame): data frame to extract into tuples
   team_id (int): id corresponding to team to fetch game logs for 
   year (int): year we are fetching game logs for 

Returns:
   tuples (list): list of tuples to be directly inserted into our database
"""


def get_team_log_tuples(df: pd.DataFrame, team_id: int, year: int):
    tuples = []
    for _, row in df.iterrows():
        game_log = (
            team_id,
            row["week"],
            row["day"],
            service_util.get_game_log_year(row["week"], year),
            row["rest_days"],
            row["home_team"],
            row["distance_traveled"],
            team_service.get_team_id_by_name(row["opp"]),
            row["result"],
            row["points_for"],
            row["points_allowed"],
            row["tot_yds"],
            row["pass_yds"],
            row["rush_yds"],
            row["opp_tot_yds"],
            row["opp_pass_yds"],
            row["opp_rush_yds"],
        )
        tuples.append(game_log)

    return tuples


"""
Utility function to retrieve a teams ID based on their name 

Args:
   team_name (str): team name to retrieve ID for 
   teams_and_ids (list): list of dictionaries containing team names and ids 

Returns: 
   id (int): ID corresponding to team name
"""


def get_team_id_by_name(team_name: str, teams_and_ids: list):
    team = next((team for team in teams_and_ids if team["name"] == team_name), None)
    return team["team_id"] if team else None


"""
Functionality to determine if a game log was persisted for a given week 

Args:
   game_log_pk (dict): PK to check is persisted in DB 

Returns:
   game_log (dict): None or persisted game log
"""


def is_game_log_persisted(game_log_pk: dict):
    game_log = fetch_data.fetch_team_game_log_by_pk(game_log_pk)

    if game_log == None:
        return False
    else:
        return True


"""
Functionality to retrieve all game logs for a particular season 

Args:
   team_id (int): team ID to fetch game logs for
   year (int): year to fetch game logs for 

Returns:
   game_logs (list): list of game logs corresponding to team
"""


def get_teams_game_logs_for_season(team_id: int, year: int):
    logging.info(
        f"Fetching all game logs for the following team ID: {team_id} and year: {year}"
    )
    return fetch_data.fetch_all_teams_game_logs_for_season(team_id, year)


"""
Utility function to determine if a teams game log has previously been inserted 

Args: 
   team_metrics (list): list of dictionaries containing team's name & game logs 
   curr_year (int): current year
   teams_and_ids (list): list of dictionaries containing team's name & corresponding team ID 
   
Returns:
   None
"""


def remove_previously_inserted_game_logs(team_metrics, curr_year, teams_and_ids):
    team_metric_pks = []

    # generate pks for each team game log
    for team in team_metrics:
        df = team["team_metrics"]
        if len(df) == 1:
            week = str(df.iloc[0]["week"])
            team_metric_pks.append(
                {
                    "team_id": get_team_id_by_name(team["team_name"], teams_and_ids),
                    "week": week,
                    "year": service_util.get_game_log_year(week, curr_year),
                }
            )

    # check if this execution is for recent games or not
    if len(team_metrics) != len(team_metric_pks):
        logging.info(
            "Program execution is not for most recent games; skipping check for previously persisted team game logs"
        )
        return

    # remove duplicate entires
    index = 0
    while index < len(team_metrics):
        if is_game_log_persisted(team_metric_pks[index]):
            del team_metrics[index]
            del team_metric_pks[index]
        else:
            logging.debug(
                f"Team game log corresponding to PK [{team_metric_pks[index]}] not persisted; inserting new game log"
            )
            index += 1


"""
Functionality to calculate rankings (offense & defense) for a team

Args:
   curr_year (int): year to take into account when fetching rankings 

Returns 
   None 
"""


def calculate_all_teams_rankings(curr_year: int):
    logging.info(
        f"Attemtping to calculate teams off/def rankings based on metrics for {curr_year} season"
    )

    # fetch all teams
    teams = team_service.get_all_teams()

    # accumulate relevant metrics (off/def) for given season for each team
    logging.info(
        f"Aggergating relevant offensive and defensive metrics for the following season: {curr_year}"
    )
    teams_metrics = [
        get_aggregate_season_metrics(
            get_teams_game_logs_for_season(team.get("team_id"), curr_year)
        )
        for team in teams
    ]

    # calculate rankings
    off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks = calculate_rankings(
        teams_metrics
    )

    logging.info(f"Offense Rush Ranks: {off_rush_ranks}\n\n")
    logging.info(f"Offense Pass Ranks: {off_pass_ranks}\n\n")
    logging.info(f"Defense Rush Ranks: {def_rush_ranks}\n\n")
    logging.info(f"Defense Pass Ranks: {def_pass_ranks}\n\n")

    # persist rankings
    update_teams_rankings(
        off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks
    )


"""
Functionality to calculate the rankings of teams based on relevant metric accumulations

Args:
   team_game_logs (list): list of game logs for a particular team 

Returns:
   off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks (tuple): tuple containing teams respective rankings 
"""


def calculate_rankings(metrics: list):
    logging.info(
        "Attempting to calculate offensive/defensive rush/pass rankings for each team based on metrics"
    )
    (
        off_rush_weighted_sums,
        def_rush_weighted_sums,
        off_pass_weighted_sums,
        def_pass_weighted_sums,
    ) = ([], [], [], [])

    # normalize metrics & apply corresponding weights
    weighted_metrics = normalize_metrics_and_apply_weights(metrics)

    logging.info(
        "Creating lists with weighted sums for each relevant metric: off_rushing, off_passing, def_rushing, and def_passing"
    )
    for metric in weighted_metrics:
        off_rush_weighted_sums.append(
            {
                "team_id": metric["team_id"],
                "total": metric["points_for"] + metric["rush_yards_for"],
            }
        )
        def_rush_weighted_sums.append(
            {
                "team_id": metric["team_id"],
                "total": metric["points_against"] + metric["rush_yards_against"],
            }
        )
        off_pass_weighted_sums.append(
            {
                "team_id": metric["team_id"],
                "total": metric["points_for"] + metric["pass_yards_for"],
            }
        )
        def_pass_weighted_sums.append(
            {
                "team_id": metric["team_id"],
                "total": metric["points_against"] + metric["pass_yards_against"],
            }
        )

    # sort each list by their total (lower indicies are higher ranks)
    logging.info(
        "Sorting weighted sums in ascending order for defenenses, and descending order for offenses (lower index, higher rank)"
    )
    off_rush_weighted_sums = sorted(
        off_rush_weighted_sums, key=lambda x: x["total"], reverse=True
    )
    def_rush_weighted_sums = sorted(
        def_rush_weighted_sums, key=lambda x: x["total"], reverse=False
    )
    off_pass_weighted_sums = sorted(
        off_pass_weighted_sums, key=lambda x: x["total"], reverse=True
    )
    def_pass_weighted_sums = sorted(
        def_pass_weighted_sums, key=lambda x: x["total"], reverse=False
    )

    # get team rankings
    logging.info(
        "Calculating team ranks based on corresponding order in each respective list [off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks]"
    )
    off_rush_ranks = [
        {"team_id": value["team_id"], "rank": index + 1}
        for index, value in enumerate(off_rush_weighted_sums)
    ]
    off_pass_ranks = [
        {"team_id": value["team_id"], "rank": index + 1}
        for index, value in enumerate(off_pass_weighted_sums)
    ]
    def_rush_ranks = [
        {"team_id": value["team_id"], "rank": index + 1}
        for index, value in enumerate(def_rush_weighted_sums)
    ]
    def_pass_ranks = [
        {"team_id": value["team_id"], "rank": index + 1}
        for index, value in enumerate(def_pass_weighted_sums)
    ]

    return off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks


"""
Determine rankings based on off/def for each team 

Args:
   weighted_sums (list): list of dictionary items sorted by their weighted sums (lower indicies, higher rank)

Returns: 
   
"""


def get_rankings(weighted_sums: list):
    ranks = [
        {"team_id": value["team_id"], "rank": index}
        for index, value in enumerate(weighted_sums)
    ]
    return ranks


"""
Obtain relevant offensive & defensive aggergated metrics (total tds, total pass yds, total rush_yds) for a team 

Args:
   team_game_logs (list): list of team game logs to obtain metrics for 

Returns:
   metrics (dict): dictionary containing following information
      team_id, points_for, points_against, pass_yards_for, pass_yards_against, rush_yards_for, rush_yards_against 
"""


def get_aggregate_season_metrics(team_game_logs: list):
    # ensure game logs passed
    if not team_game_logs:
        logging.error("Unable to get relevant season metrics for an empty list")
        raise Exception(
            "Unable to obtain relevant off/def metrics because no team game logs were passed"
        )

    # metrics to account for
    points_for = 0
    points_against = 0
    pass_yards_for = 0
    pass_yards_against = 0
    rush_yards_for = 0
    rush_yards_against = 0

    for game_log in team_game_logs:
        points_for += game_log.get("points_for", 0)
        points_against += game_log.get("points_allowed", 0)
        pass_yards_for += game_log.get("pass_yds", 0)
        pass_yards_against += game_log.get("opp_pass_yds", 0)
        rush_yards_for += game_log.get("rush_yds", 0)
        rush_yards_against += game_log.get("opp_rush_yds", 0)

    return {
        "team_id": team_game_logs[0].get("team_id"),
        "points_for": points_for,
        "points_against": points_against,
        "pass_yards_for": pass_yards_for,
        "pass_yards_against": pass_yards_against,
        "rush_yards_for": rush_yards_for,
        "rush_yards_against": rush_yards_against,
    }


"""
Persists updated team rankings 

Args:
   off_rush_ranks (list): offensive rush rankings
   off_pass_ranks (list): offensive pass rankings
   def_rush_ranks (list): defensive rush rankings
   def_pass_ranks (list): defensive pass rankings

Returns:
   None
"""


def update_teams_rankings(
    off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks
):
    logging.info(
        "Attempting to update team records with updated off/def passing/rushing rankings"
    )
    insert_data.update_team_rankings(off_rush_ranks, "off_rush_rank")
    insert_data.update_team_rankings(off_pass_ranks, "off_pass_rank")
    insert_data.update_team_rankings(def_pass_ranks, "def_pass_rank")
    insert_data.update_team_rankings(def_rush_ranks, "def_rush_rank")


"""
Functionality to normalize players season metrics, ensuring metrics 
like passing_yds & total points are on the same scale

Args:
   team_metrics (dict):  relevant in season metrics for a team

Returns:
   normalized (dict): metric with normalized values 
"""


def normalize_metrics_and_apply_weights(team_metrics: list):
    logging.info(
        f"Attempting to normalize relevant in season metrics and apply corresponding weights based on configurations"
    )
    keys = [
        "points_for",
        "points_against",
        "rush_yards_for",
        "rush_yards_against",
        "pass_yards_for",
        "pass_yards_against",
    ]

    # inialize min/max values with first values
    min_metrics = {key: team_metrics[0].get(key) for key in keys}
    max_metrics = {key: team_metrics[0].get(key) for key in keys}

    # obtain min/max values for each respective key
    for team in team_metrics[1:]:
        for key in keys:
            value = team[key]
            min_metrics[key] = min(value, min_metrics[key])
            max_metrics[key] = max(value, max_metrics[key])

    # normalize metrics & apply weights
    weighted_metrics = []
    for team in team_metrics:
        for key in keys:
            curr_val = team[key]
            max_value = max_metrics[key]
            min_value = min_metrics[key]

            if max_value - min_value > 0:
                normalized_value = (curr_val - min_value) / (max_value - min_value)
            else:
                normalized_value = 0
            team[key] = normalized_value

        # apply weights
        weighted_metrics.append(apply_weights(team, keys, team["team_id"]))

    return weighted_metrics


"""
Apply weights determined in configurations to normalized aggregate season metrics in order to properly rank 

TODO (FFM-129): Optimize Points_For & Points_Against to account for where points came from when applying weights (i.e passing tds/rushing tds/def tds)

Args:
   normalized_metrics (dict): normalized metrics to apply weights to 
   keys (list): list of keys corresponding to metrics
   team_id (int): id corresponding to team metrics apply to

Returns: 
   weighted_metrics (list): metrics with weights applied 
"""


def apply_weights(normalized_metrics: dict, keys: list, team_id: int):
    yd_weight = props.get_config("rankings.weights.yards")
    td_weight = props.get_config("rankings.weights.td")

    weights = {
        "points_for": td_weight,
        "points_against": td_weight,
        "rush_yards_for": yd_weight,
        "rush_yards_against": yd_weight,
        "pass_yards_for": yd_weight,
        "pass_yards_against": yd_weight,
    }

    weighted_metrics = {key: normalized_metrics[key] * weights[key] for key in keys}
    weighted_metrics["team_id"] = team_id  # add team id to dict

    logging.info(f"Weighted metrics: [{weighted_metrics}]\n\n")
    return weighted_metrics
