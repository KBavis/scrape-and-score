import logging
import pandas as pd
from scrape_and_score.service import team_service
from scrape_and_score.config import props
from scrape_and_score.db.read.teams import (
    fetch_team_game_log_by_pk,
    fetch_all_teams_game_logs_for_season,
    fetch_pks_for_inserted_team_game_logs,
    fetch_max_week_rankings_calculated_for_season,
    check_bye_week_rankings_exists,
)
from scrape_and_score.db.insert.teams import (
    update_team_game_logs,
    insert_team_game_logs,
    insert_bye_week_rankings,
    insert_team_rankings,
)


def insert_multiple_teams_game_logs(
    team_game_logs: list,
    teams_and_ids: list,
    year: int = None,
    should_update: bool = False,
):
    """
    Functionality to insert multiple teams game logs

    Args:
        team_game_logs (list): list of dictionaries containing team's name & game logs
        teams_and_ids (list): list of dictionaries containing team's name & corresponding team ID
        should_update (bool): flag to indicate if we want to update previously persisted game logs or not
    """

    # extract or remove previously inserted game logs
    update_game_logs, insert_game_logs = filter_previously_inserted_game_logs(
        team_game_logs, teams_and_ids, year, should_update
    )
    logging.info(
        f"Filtering complete; planning on updating {len(update_game_logs)} records and inserting {len(insert_game_logs)} records."
    )

    if len(update_game_logs) == 0 and len(insert_game_logs) == 0:
        logging.info(
            "No new team game logs to persist and no new game logs to update; skipping updates/insertions"
        )
        return

    # update & insert game logs if necessary
    if update_game_logs:
        insert_or_update_team_game_logs(update_game_logs, teams_and_ids, year, True)

    if insert_game_logs:
        insert_or_update_team_game_logs(insert_game_logs, teams_and_ids, year, False)


def insert_or_update_team_game_logs(
    team_game_logs: list, teams_and_ids: list, year: int, is_update: bool
):
    """
    Insert or update game logs

    Args:
        team_game_logs (list): the team game logs to update or insert
        teams_and_ids (list): teams and corresponding team ids
        year (int) : season game logs correspond to
        is_update (bool): flag to indicate if we are updating or inserting
    """

    # insert game logs
    for team in team_game_logs:
        team_name = team["team_name"]
        logging.info(
            f"Attempting to {'insert' if not is_update else 'update'} game logs for team '{team_name}"
        )

        df = team["team_metrics"]
        if df.empty:
            logging.warning(
                f"No team metrics corresponding to team '{team_name}; skipping insertion"
            )
            continue

        # fetch team ID
        team_id = get_team_id_by_name(team_name, teams_and_ids)

        # insert team game logs into db
        if is_update:
            tuples = get_team_log_tuples_for_update(df, team_id, year)
            update_team_game_logs(tuples)
        else:
            tuples = get_insert_team_log_tuples(df, team_id, year)
            insert_team_game_logs(tuples)


def get_insert_team_log_tuples(df: pd.DataFrame, team_id: int, year: int):
    """
    Utility function to fetch team game log tuples to insert into our database

    Args:
        df (pd.DataFrame): data frame to extract into tuples
        team_id (int): id corresponding to team to fetch game logs for
        year (int): year we are fetching game logs for

    Returns:
        list: list of tuples to be directly inserted into our database
    """

    tuples = []

    opponent_map = {
        "Oakland Raiders": "Las Vegas Raiders",
        "Washington Redskins": "Washington Commanders",
        "Washington Football Team": "Washington Commanders",
        "San Diego Chargers": "Los Angeles Chargers",
    }

    for _, row in df.iterrows():
        opponent = opponent_map.get(row["opp"], row["opp"])
        game_log = (
            team_id,
            row["week"],
            row["day"],
            year,
            row["rest_days"],
            row["home_team"],
            row["distance_traveled"],
            team_service.get_team_id_by_name(opponent),
            row["result"],
            row["points_for"],
            row["points_allowed"],
            row["tot_yds"],
            row["pass_yds"],
            row["rush_yds"],
            row["pass_tds"],
            row["pass_cmp"],
            row["pass_att"],
            row["pass_cmp_pct"],
            row["rush_att"],
            row["rush_tds"],
            row["yds_gained_per_pass_att"],
            row["adj_yds_gained_per_pass_att"],
            row["pass_rate"],
            row["sacked"],
            row["sack_yds_lost"],
            row["rush_yds_per_att"],
            row["total_off_plays"],
            row["yds_per_play"],
            row["fga"],
            row["fgm"],
            row["xpa"],
            row["xpm"],
            row["total_punts"],
            row["punt_yds"],
            row["pass_fds"],
            row["rsh_fds"],
            row["pen_fds"],
            row["total_fds"],
            row["thrd_down_conv"],
            row["thrd_down_att"],
            row["fourth_down_conv"],
            row["fourth_down_att"],
            row["penalties"],
            row["penalty_yds"],
            row["fmbl_lost"],
            row["interceptions"],
            row["turnovers"],
            row["time_of_poss"],
        )

        tuples.append(game_log)

    return tuples


def get_team_log_tuples_for_update(df: pd.DataFrame, team_id: int, year: int):
    """
    Extract reelvant team game log tuples required for updating DB entries

    Args:
        df (pd.DataFrame): data frame containing relevant data
        team_id (int): team ID that corresponds to relevant team
        year (int): relevant year
    """

    tuples = []

    opponent_map = {
        "Oakland Raiders": "Las Vegas Raiders",
        "Washington Redskins": "Washington Commanders",
        "Washington Football Team": "Washington Commanders",
        "San Diego Chargers": "Los Angeles Chargers",
    }

    for _, row in df.iterrows():
        opponent = opponent_map.get(row["opp"], row["opp"])

        game_log = (
            row["day"],
            row["rest_days"],
            row["home_team"],
            row["distance_traveled"],
            team_service.get_team_id_by_name(opponent),
            row["result"],
            row["points_for"],
            row["points_allowed"],
            row["tot_yds"],
            row["pass_yds"],
            row["rush_yds"],
            row["pass_tds"],
            row["pass_cmp"],
            row["pass_att"],
            row["pass_cmp_pct"],
            row["rush_att"],
            row["rush_tds"],
            row["yds_gained_per_pass_att"],
            row["adj_yds_gained_per_pass_att"],
            row["pass_rate"],
            row["sacked"],
            row["sack_yds_lost"],
            row["rush_yds_per_att"],
            row["total_off_plays"],
            row["yds_per_play"],
            row["fga"],
            row["fgm"],
            row["xpa"],
            row["xpm"],
            row["total_punts"],
            row["punt_yds"],
            row["pass_fds"],
            row["rsh_fds"],
            row["pen_fds"],
            row["total_fds"],
            row["thrd_down_conv"],
            row["thrd_down_att"],
            row["fourth_down_conv"],
            row["fourth_down_att"],
            row["penalties"],
            row["penalty_yds"],
            row["fmbl_lost"],
            row["interceptions"],
            row["turnovers"],
            row["time_of_poss"],
            team_id,
            row["week"],
            year,
        )

        tuples.append(game_log)

    return tuples


def get_team_id_by_name(team_name: str, teams_and_ids: list):
    """
    Utility function to retrieve a teams ID based on their name

    Args:
        team_name (str): team name to retrieve ID for
        teams_and_ids (list): list of dictionaries containing team names and ids

    Returns:
        id (int): ID corresponding to team name
    """

    team = next((team for team in teams_and_ids if team["name"] == team_name), None)
    return team["team_id"] if team else None


def is_game_log_persisted(game_log_pk: dict):
    """
    Functionality to determine if a game log was persisted for a given week

    Args:
        game_log_pk (dict): PK to check is persisted in DB

    Returns:
        game_log (dict): None or persisted game log
    """
    game_log = fetch_team_game_log_by_pk(game_log_pk)

    if game_log == None:
        return False
    else:
        return True


def get_teams_game_logs_for_season(team_id: int, year: int):
    """
    Functionality to retrieve all game logs for a particular season

    Args:
        team_id (int): team ID to fetch game logs for
        year (int): year to fetch game logs for

    Returns:
        game_logs (list): list of game logs corresponding to team
    """
    logging.info(
        f"Fetching all game logs for the following team ID: {team_id} and year: {year}"
    )
    return fetch_all_teams_game_logs_for_season(team_id, year)


def filter_previously_inserted_game_logs(
    team_game_logs: list, teams_and_ids: list, year: int, should_update: bool
):
    """
    Utility function to determine if a teams game log has previously been inserted

    Args:
        team_metrics (list): list of dictionaries containing team's name & game logs
        teams_and_ids (list): list of dictionaries containing team's name & corresponding team ID
        year (int): season pertaining to game log
        should_update (bool): boolean indicating if we should not remove game logs and instead seperate them
    """

    # generate pks for each team game log
    persisted_team_game_log_pks = fetch_pks_for_inserted_team_game_logs(year)

    update_game_logs = []
    insert_game_logs = []

    index = 0
    while index < len(team_game_logs):
        game_log = team_game_logs[index]
        pk = {
            "team_id": get_team_id_by_name(game_log["team_name"], teams_and_ids),
            "week": int(game_log["team_metrics"].iloc[0]["week"]),
            "year": year,
        }

        print(pk)

        if pk in persisted_team_game_log_pks:
            # only account for previously persisted if planning on updating
            if should_update:
                update_game_logs.append(game_log)
        else:
            insert_game_logs.append(game_log)

        index += 1

    return update_game_logs, insert_game_logs


def calculate_all_teams_rankings(year: int, week: int = None):
    """
    Functionality to calculate rankings (offense & defense) for a team

    Args:
        year (int): year to take into account when fetching rankings
        week (int): optional week parameter
    """

    logging.info(
        f"Attemtping to calculate teams off/def rankings based on metrics for {year} season"
    )

    # fetch all teams
    teams = team_service.get_all_teams()

    # accumulate relevant metrics (off/def) for given season for each team
    logging.info(
        f"Aggergating relevant offensive and defensive metrics for the following season: {year}"
    )

    # fetch all game logs for current season
    teams_game_logs = [
        get_teams_game_logs_for_season(team.get("team_id"), year) for team in teams
    ]

    # filter game logs by week if necessary
    if week is not None:
        filtered_game_logs = []

        for game_log in teams_game_logs:
            if game_log["week"] == week:
                filtered_game_logs.append(game_log)

        if not filtered_game_logs:
            raise Exception(
                f"Unable to retireve team game logs corresponding to Week {week} of the {year} NFL Season to calculate team weekly rankings"
            )

        teams_game_logs = filtered_game_logs

    # calculate weekly aggregate metrics
    teams_weekly_aggregate_metrics = [
        get_weekly_aggergate_metrics(season_game_logs)
        for season_game_logs in teams_game_logs
    ]

    # calculate rankings up to max week
    calculate_and_persist_weekly_rankings(teams_weekly_aggregate_metrics, year, week)


def calculate_and_persist_weekly_rankings(
    teams_weekly_aggregate_metrics: list, season: int, rel_week: int
):
    """
    Function to calculate the weekly rankings for a team

    Args:
        teams_weekly_aggregate_metrics (list): list of season long aggregate metrics for a team
        season (int): season to calculate weekly rankings for
        week (int): relvant week

    Returns:
        weekly_rankings (list): weekly rankings that need to be persisted
    """

    # isnert bye week rankings
    insert_bye_week_rankings(teams_weekly_aggregate_metrics, season)

    # determine max week we have metrics for
    max_week = max(
        metrics["week"]
        for team_metrics in teams_weekly_aggregate_metrics
        for metrics in team_metrics
    )

    # determine week to start calculating metrics for
    max_persisted_week = fetch_max_week_rankings_calculated_for_season(season)
    curr_week = 1 if max_persisted_week is None else max_persisted_week + 1

    if curr_week >= max_week:
        logging.info(
            f"All team rankings are persisted for the season {season}; skipping calculating & persisting of team rankings"
        )

    # utilize only reelvant week if necessary
    weeks = (
        range(curr_week, max_week + 1)
        if rel_week is None
        else range(rel_week, rel_week + 1)
    )

    # loop through relevant weeks
    for week in weeks:

        # extract each teams cumulative rankings corresponding to current week
        teams_curr_week_agg_metrics = []
        for team_metrics in teams_weekly_aggregate_metrics:
            week_metrics = [
                metrics for metrics in team_metrics if metrics["week"] == week
            ]
            teams_curr_week_agg_metrics.extend(week_metrics)

        # calculate rankings
        off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks = (
            calculate_rankings(teams_curr_week_agg_metrics)
        )

        team_ranks_records = generate_team_ranks_records(
            off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks, week, season
        )

        insert_teams_ranks(team_ranks_records)


def generate_team_ranks_records(
    off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks, week, season
):
    """
    Functionality to generate proper records needed to insert into our database

    Args:
        off_rush_ranks (list): team_ids and corresponding ranks
        off_pass_ranks (list): team_ids and corresponding ranks
        def_rush_ranks (list): team_ids and corresponding ranks
        def_pass_ranks (list): team_ids and corresponding ranks
        week (int): week we are accounting for
        season (int): season we are accounting for
    """

    team_ids = [team.get("team_id") for team in off_rush_ranks]

    records = []
    for team_id in team_ids:
        record = {}

        record["off_rush_rank"] = next(
            team["rank"] for team in off_rush_ranks if team["team_id"] == team_id
        )
        record["off_pass_rank"] = next(
            team["rank"] for team in off_pass_ranks if team["team_id"] == team_id
        )
        record["def_pass_rank"] = next(
            team["rank"] for team in def_pass_ranks if team["team_id"] == team_id
        )
        record["def_rush_rank"] = next(
            team["rank"] for team in def_rush_ranks if team["team_id"] == team_id
        )
        record["team_id"] = team_id
        record["season"] = season
        record["week"] = week

        records.append(record)

    return records


def insert_bye_week_rankings(teams_weekly_aggregate_metrics: list, season: int):
    """
    Determine teams bye week rankings

    Args:
        teams_weekly_aggregate_metrics (list): list of teams accumualted weekly metrics
    """

    logging.info(f"Attempting to insert bye week rankings for the season: {season}")

    team_bye_weeks = []
    for curr_team_weekly_metrics in teams_weekly_aggregate_metrics:

        # skip teams with bye weeks already persisted for given season
        bye_week = check_bye_week_rankings_exists(
            curr_team_weekly_metrics[0]["team_id"], season
        )
        if bye_week is not None:
            continue

        # init first metric and prev_week
        metric = curr_team_weekly_metrics[0]
        prev_week = metric["week"]

        for metric in curr_team_weekly_metrics[1:]:
            curr_week = metric["week"]

            # check if bye week
            if curr_week - prev_week > 1:
                bye_week = curr_week - 1
                break

            prev_week = curr_week

        team_bye = {"team_id": metric["team_id"], "week": bye_week}
        team_bye_weeks.append(team_bye)

    # insert bye week rankings into DB
    if len(team_bye_weeks) == 0:
        logging.info(
            f"All bye weeks for teams in the season {season} have already been inserted into our DB; skipping insertion"
        )
        return
    insert_bye_week_rankings(team_bye_weeks, season)


def get_weekly_aggergate_metrics(season_game_logs: list):
    """
    Helper function to calculate the weekly aggergate metrics of a teams season game log metrics

    Args:
        season_game_logs (list): list of season long game logs

    Returns:
        list: weekly aggregate metrics
    """

    # metrics to account for
    points_for = 0
    points_against = 0
    pass_yards_for = 0
    pass_yards_against = 0
    rush_yards_for = 0
    rush_yards_against = 0

    weekly_aggregate_metrics = []

    for weekly_game_log in season_game_logs:

        points_for += weekly_game_log.get("points_for", 0)
        points_against += weekly_game_log.get("points_allowed", 0)
        pass_yards_for += weekly_game_log.get("pass_yds", 0)
        pass_yards_against += weekly_game_log.get("opp_pass_yds", 0)
        rush_yards_for += weekly_game_log.get("rush_yds", 0)
        rush_yards_against += weekly_game_log.get("opp_rush_yds", 0)

        metrics = {
            "team_id": weekly_game_log.get("team_id"),
            "week": weekly_game_log.get("week"),
            "points_for": points_for,
            "points_against": points_against,
            "pass_yards_for": pass_yards_for,
            "pass_yards_against": pass_yards_against,
            "rush_yards_for": rush_yards_for,
            "rush_yards_against": rush_yards_against,
        }
        weekly_aggregate_metrics.append(metrics)

    return weekly_aggregate_metrics


def calculate_rankings(metrics: list):
    """
    Functionality to calculate the rankings of teams based on relevant metric accumulations

    TODO (FFM-314): Account for the break down of scoring (i.e rushing tds for, rushing tds against, etc)

    Args:
        team_game_logs (list): list of game logs for a particular team

    Returns:
        tuple: teams respective rankings
    """

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


def insert_teams_ranks(records):
    """
    Persists team rankings

    Args:
        records (list): list of team ranking records to persist
    """

    logging.info("Attempting to insert team rankings records")
    insert_team_rankings(records)


def normalize_metrics_and_apply_weights(team_metrics: list):
    """
    Functionality to normalize players season metrics, ensuring metrics
    like passing_yds & total points are on the same scale

    Args:
        team_metrics (dict):  relevant in season metrics for a team

    Returns:
        normalized (dict): metric with normalized values
    """

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


def apply_weights(normalized_metrics: dict, keys: list, team_id: int):
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
