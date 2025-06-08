import logging
from scrape_and_score.db.connection import get_connection


def fetch_all_teams():
    """
    Functionality to fetch all teams persisted in database
    """

    sql = "SELECT * FROM team"
    teams = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

            for row in rows:
                teams.append({"team_id": row[0], "name": row[1]})

    except Exception as e:
        logging.error("An error occurred while fetching all teams: {e}")
        raise e

    return teams


def fetch_game_date_from_team_game_log(season: int, week: int, team_id: int):
    """
    Retrieve game data corresponding to team game log

    Args:
        season (int): relevant season
        week (int): relevant week
        team_id (int): relevant team
    """

    sql = "SELECT game_date FROM team_game_log WHERE team_id = %s AND year = %s AND week = %s"
    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (team_id, season, week))
            row = cur.fetchone()

            if row:
                return row[0]
            else:
                return None

    except Exception as e:
        logging.error(
            "An error occurred while fetching game date for previous week",
            exc_info=True,
        )
        raise e


def fetch_team_by_name(team_name: int):
    """
    Functionality to fetch a team by their team name

    Args:
        team_name (str): team name to retrieve team by

    Returns:
        team (dict): team record
    """

    sql = "SELECT * FROM team WHERE name = %s"
    team = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (team_name,))  # ensure team_name in tuple
            row = cur.fetchone()

            if row:
                team = {"team_id": row[0], "name": row[1]}

    except Exception as e:
        logging.error(
            f"An error occurred while fetching team with name {team_name}: {e}"
        )
        raise e

    return team


def fetch_team_name_by_id(id: int):
    """
    Functionality to retrieve team name by ID

    Args:
        id (int): id to retrieve name for
    """

    sql = "SELECT name FROM team WHERE team_id = %s"

    name = None
    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (id,))
            row = cur.fetchone()

            if row:
                name = row[0]

    except Exception as e:
        logging.error(f"An error occurred while fetching team with team_id[{id}]: {e}")
        raise e

    return name


def fetch_team_seasonal_metrics(team_id: int, season: int):
    """
    Retrieve team seasonal metrics for a particular season

    Args:
        team_id (int): relevant team
        season (int): relevant season
    """

    sql = """
        SELECT 
            *
        FROM team_seasonal_general_metrics g
        LEFT JOIN team_seasonal_kicking_metrics k ON g.team_id = k.team_id AND g.season = k.season
        LEFT JOIN team_seasonal_punting_metrics pu ON g.team_id = pu.team_id AND g.season = pu.season
        LEFT JOIN team_seasonal_passing_metrics pa ON g.team_id = pa.team_id AND g.season = pa.season
        LEFT JOIN team_seasonal_rushing_receiving_metrics rr ON g.team_id = rr.team_id AND g.season = rr.season
        LEFT JOIN team_seasonal_defensive_metrics d ON g.team_id = d.team_id AND g.season = d.season
        LEFT JOIN team_seasonal_scoring_metrics s ON g.team_id = s.team_id AND g.season = s.season
        LEFT JOIN team_seasonal_ranks r ON g.team_id = r.team_id AND g.season = r.season
        WHERE g.team_id = %s AND g.season = %s
    """

    metrics = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (team_id, season))
            row = cur.fetchone()

            if row:
                colnames = [desc[0] for desc in cur.description]
                metrics = dict(zip(colnames, row))
            else:
                logging.warning(
                    f"No metrics found for team_id={team_id}, season={season}"
                )

    except Exception as e:
        logging.error(
            f"An error occurred while fetching team metrics for team_id={team_id}, season={season}: {e}"
        )
        raise e

    return metrics


def fetch_game_conditions_record_by_pk(pk: dict):
    """
    Retrieve game_conditions record by its composite primary key.

    Args:
        pk (dict): Dictionary with keys 'season', 'week', 'home_team_id', 'visit_team_id'

    Returns:
        dict or None: The matching game_conditions record as a dictionary, or None if not found
    """

    sql = """
        SELECT 
            game_date,
            game_time,
            kickoff,
            month,
            start,
            surface,
            weather_icon,
            temperature,
            precip_probability,
            precip_type,
            wind_speed,
            wind_bearing
        FROM game_conditions
        WHERE season = %s AND week = %s AND home_team_id = %s AND visit_team_id = %s
    """

    try:
        connection = get_connection()
        with connection.cursor() as cur:
            cur.execute(
                sql, (pk["season"], pk["week"], pk["home_team_id"], pk["visit_team_id"])
            )
            row = cur.fetchone()

            if row:
                columns = [
                    "game_date",
                    "game_time",
                    "kickoff",
                    "month",
                    "start",
                    "surface",
                    "weather_icon",
                    "temperature",
                    "precip_probability",
                    "precip_type",
                    "wind_speed",
                    "wind_bearing",
                ]
                return dict(zip(columns, row))
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while fetching game_conditions for PK(season={pk['season']}, week={pk['week']}, "
            f"home_team_id={pk['home_team_id']}, visit_team_id={pk['visit_team_id']})",
            exc_info=True,
        )
        raise e


def fetch_team_game_logs_by_week_and_season(season: int, week: int):
    """
    Retrieve 'team_game_log' records corresponding to particular season / week

    Args:
        season (int): relevant NFL season
        week (int): relevant week

    Returns:
        list: persisted team game log records corresponding to week / season
    """

    sql = """
        SELECT
            team_id, week, day, year, rest_days, home_team, distance_traveled,
            opp, result, points_for, points_allowed, tot_yds, pass_yds, rush_yds,
            opp_tot_yds, opp_pass_yds, opp_rush_yds, pass_tds, pass_cmp, pass_att,
            pass_cmp_pct, rush_att, rush_tds, yds_gained_per_pass_att,
            adj_yds_gained_per_pass_att, pass_rate, sacked, sack_yds_lost,
            rush_yds_per_att, total_off_plays, total_yds, yds_per_play,
            fga, fgm, xpa, xpm, total_punts, punt_yds,
            pass_fds, rsh_fds, pen_fds, total_fds,
            thrd_down_conv, thrd_down_att, fourth_down_conv, fourth_down_att,
            penalties, penalty_yds, fmbl_lost, int, turnovers, time_of_poss,
            game_date
        FROM team_game_log
        WHERE week = %s AND year = %s
    """
    team_game_logs = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (week, season))
            rows = cur.fetchall()

            if not rows:
                logging.warning(
                    f"No team game logs persisted corresponding to week {week} in the {season} NFL season."
                )
                return team_game_logs

            for row in rows:
                team_game_logs.append(
                    {
                        "team_id": row[0],
                        "week": row[1],
                        "day": row[2],
                        "year": row[3],
                        "rest_days": row[4],
                        "home_team": row[5],
                        "distance_traveled": row[6],
                        "opp": row[7],
                        "result": row[8],
                        "points_for": row[9],
                        "points_allowed": row[10],
                        "tot_yds": row[11],
                        "pass_yds": row[12],
                        "rush_yds": row[13],
                        "opp_tot_yds": row[14],
                        "opp_pass_yds": row[15],
                        "opp_rush_yds": row[16],
                        "pass_tds": row[17],
                        "pass_cmp": row[18],
                        "pass_att": row[19],
                        "pass_cmp_pct": row[20],
                        "rush_att": row[21],
                        "rush_tds": row[22],
                        "yds_gained_per_pass_att": row[23],
                        "adj_yds_gained_per_pass_att": row[24],
                        "pass_rate": row[25],
                        "sacked": row[26],
                        "sack_yds_lost": row[27],
                        "rush_yds_per_att": row[28],
                        "total_off_plays": row[29],
                        "total_yds": row[30],
                        "yds_per_play": row[31],
                        "fga": row[32],
                        "fgm": row[33],
                        "xpa": row[34],
                        "xpm": row[35],
                        "total_punts": row[36],
                        "punt_yds": row[37],
                        "pass_fds": row[38],
                        "rsh_fds": row[39],
                        "pen_fds": row[40],
                        "total_fds": row[41],
                        "thrd_down_conv": row[42],
                        "thrd_down_att": row[43],
                        "fourth_down_conv": row[44],
                        "fourth_down_att": row[45],
                        "penalties": row[46],
                        "penalty_yds": row[47],
                        "fmbl_lost": row[48],
                        "int": row[49],
                        "turnovers": row[50],
                        "time_of_poss": row[51],
                        "game_date": row[52],
                    }
                )

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the team game log corresponding to PK {pk}: {e}"
        )
        raise e

    return team_game_logs


def fetch_team_game_log_by_pk(pk: dict):
    """
    Retrieve 'team_game_log' record by its PK

    Args:
        pk (dict): primary key of team game log

    Returns:
        dict: persisted DB value or None
    """

    sql = """
        SELECT
            team_id, week, day, year, rest_days, home_team, distance_traveled,
            opp, result, points_for, points_allowed, tot_yds, pass_yds, rush_yds,
            opp_tot_yds, opp_pass_yds, opp_rush_yds, pass_tds, pass_cmp, pass_att,
            pass_cmp_pct, rush_att, rush_tds, yds_gained_per_pass_att,
            adj_yds_gained_per_pass_att, pass_rate, sacked, sack_yds_lost,
            rush_yds_per_att, total_off_plays, total_yds, yds_per_play,
            fga, fgm, xpa, xpm, total_punts, punt_yds,
            pass_fds, rsh_fds, pen_fds, total_fds,
            thrd_down_conv, thrd_down_att, fourth_down_conv, fourth_down_att,
            penalties, penalty_yds, fmbl_lost, int, turnovers, time_of_poss,
            game_date
        FROM team_game_log
        WHERE team_id = %s AND week = %s AND year = %s
    """
    team_game_log = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (pk["team_id"], pk["week"], pk["year"]))
            row = cur.fetchone()

            if row:
                # Mapping columns to dictionary
                team_game_log = {
                    "team_id": row[0],
                    "week": row[1],
                    "day": row[2],
                    "year": row[3],
                    "rest_days": row[4],
                    "home_team": row[5],
                    "distance_traveled": row[6],
                    "opp": row[7],
                    "result": row[8],
                    "points_for": row[9],
                    "points_allowed": row[10],
                    "tot_yds": row[11],
                    "pass_yds": row[12],
                    "rush_yds": row[13],
                    "opp_tot_yds": row[14],
                    "opp_pass_yds": row[15],
                    "opp_rush_yds": row[16],
                    "pass_tds": row[17],
                    "pass_cmp": row[18],
                    "pass_att": row[19],
                    "pass_cmp_pct": row[20],
                    "rush_att": row[21],
                    "rush_tds": row[22],
                    "yds_gained_per_pass_att": row[23],
                    "adj_yds_gained_per_pass_att": row[24],
                    "pass_rate": row[25],
                    "sacked": row[26],
                    "sack_yds_lost": row[27],
                    "rush_yds_per_att": row[28],
                    "total_off_plays": row[29],
                    "total_yds": row[30],
                    "yds_per_play": row[31],
                    "fga": row[32],
                    "fgm": row[33],
                    "xpa": row[34],
                    "xpm": row[35],
                    "total_punts": row[36],
                    "punt_yds": row[37],
                    "pass_fds": row[38],
                    "rsh_fds": row[39],
                    "pen_fds": row[40],
                    "total_fds": row[41],
                    "thrd_down_conv": row[42],
                    "thrd_down_att": row[43],
                    "fourth_down_conv": row[44],
                    "fourth_down_att": row[45],
                    "penalties": row[46],
                    "penalty_yds": row[47],
                    "fmbl_lost": row[48],
                    "int": row[49],
                    "turnovers": row[50],
                    "time_of_poss": row[51],
                    "game_date": row[52],
                }

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the team game log corresponding to PK {pk}: {e}"
        )
        raise e

    return team_game_log


def fetch_all_teams_game_logs_for_season(team_id: int, year: int):
    """
    Functionality to retrieve teams game logs for a particular season

    Args:
        year (int): year to fetch team game logs for
        team_id (int): team to fetch game logs for

    Returns
        game_logs (list): list of game_logs for a particular season/team
    """

    team_game_logs = []
    sql = "SELECT * FROM team_game_log WHERE team_id = %s AND year=%s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (team_id, year))
            rows = cur.fetchall()

            for row in rows:
                team_game_log = {
                    "team_id": row[0],
                    "week": row[1],
                    "day": row[2],
                    "year": row[3],
                    "rest_days": row[4],
                    "home_team": row[5],
                    "distance_traveled": row[6],
                    "opp": row[7],
                    "result": row[8],
                    "points_for": row[9],
                    "points_allowed": row[10],
                    "tot_yds": row[11],
                    "pass_yds": row[12],
                    "rush_yds": row[13],
                    "opp_tot_yds": row[14],
                    "opp_pass_yds": row[15],
                    "opp_rush_yds": row[16],
                }

                team_game_logs.append(team_game_log)

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the team game logs corresponding to team {team_id} and year {year}: {e}"
        )
        raise e

    return team_game_logs


def fetch_max_week_persisted_in_team_betting_odds_table(year: int):
    """
    Retrieve the latest week we have persisted data in our 'team_betting_odds' table

    Args:
        year (int): season to retrieve data for

    Return:
        week (int): the latest week persisted in our db table
    """

    sql = "SELECT week FROM team_betting_odds WHERE season = %s AND week = (SELECT MAX(week) FROM team_betting_odds WHERE season = %s)"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year, year))
            row = cur.fetchone()

            if row:
                week = row[0]

                if week:
                    return int(week)

            raise Exception("Unable to extract week from team_betting_odds table")

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the latest week persisted in team_betting_odds for year {year}: {e}"
        )
        raise e


def fetch_team_betting_odds_by_pk(
    home_team_id: int, away_team_id: int, season: int, week: int
):
    """
    Retrieve 'team_betting_odds' record from DB by PK if its exists

    Args:
        home_team_id (int): team ID corresponding to home team
        away_team_id (int): team ID corresponding to away team
        season (int): relevant season
        week (int): relevant week

    Returns:
        dict: team betting odds record persisted
    """

    sql = """
        SELECT home_team_id, away_team_id, home_team_score, away_team_score, week, season, game_over_under, favorite_team_id, spread, total_points, over_hit, under_hit, favorite_covered, underdog_covered
        FROM team_betting_odds 
        WHERE season = %s AND week = %s AND home_team_id = %s AND away_team_id = %s 
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season, week, home_team_id, away_team_id))
            row = cur.fetchone()

            if row:
                return {
                    "home_team_id": row[0],
                    "away_team_id": row[1],
                    "home_team_score": row[2],
                    "away_team_score": row[3],
                    "week": row[4],
                    "season": row[5],
                    "game_over_under": row[6],
                    "favorite_team_id": row[7],
                    "spread": row[8],
                    "total_points": row[9],
                    "over_hit": row[10],
                    "under_hit": row[11],
                    "favorite_covered": row[12],
                    "underdog_covered": row[13],
                }

        return None

    except Exception as e:
        logging.error(
            f"An error occurred while fetching team betting odds record by PK",
            exc_info=True,
        )
        raise e


def check_bye_week_rankings_exists(team_id: int, season: int):
    """
    Functionality to check if the bye week rankings exists for a given team in a given season

    Args:
        team_id (int): team to check for
        season (int): season to check for

    Returns:
        bye_week (int): the bye week for given team
    """

    sql = "SELECT week FROM team_ranks WHERE team_id = %s AND season = %s AND off_rush_rank = -1 AND off_pass_rank = -1 AND def_pass_rank = -1 AND def_rush_rank = -1"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (team_id, season))
            row = cur.fetchone()

            if row:
                return row[0]
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while checking if the bye week ranking exists for the team {team_id} in the corresponding season: {season}: {e}"
        )
        raise e


def fetch_max_week_rankings_calculated_for_season(season: int):
    """
    Retrieve the max week persisted in the team_ranks table for the specified season

    Args:
        season (int): season to retrieve max week for

    Returns:
        max (int): maximum week with rank associated with it for given season
    """

    sql = "SELECT MAX(week) FROM team_ranks WHERE season = %s AND off_rush_rank != -1"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season,))
            row = cur.fetchone()

            if row:
                return row[0]
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving hte max week with rankings persisted for the corresponding season: {season}: {e}"
        )
        raise e


def fetch_teams_home_away_wins_and_losses(season: int, team_id: int):
    """
    Functionality to retrieve teams home and away wins and losses for a given season and team

    Args:
        season (int): season to retrieve data for
        team_id (int): team ID to retrieve data for

    Returns:
        dict: number of wins / losses for respective team
    """

    sql = """
        WITH record_counts AS (
            SELECT 
                COUNT(CASE WHEN result = 'W' THEN 1 END) as wins,
                COUNT(CASE WHEN result = 'L' THEN 1 END) as losses,
                COUNT(CASE WHEN home_team = true AND result = 'W' THEN 1 END) as home_wins,
                COUNT(CASE WHEN home_team = true AND result = 'L' THEN 1 END) as home_losses,
                COUNT(CASE WHEN home_team = false AND result = 'W' THEN 1 END) as away_wins,
                COUNT(CASE WHEN home_team = false AND result = 'L' THEN 1 END) as away_losses,
                CAST(COUNT(CASE WHEN result = 'W' THEN 1 END) AS FLOAT) / 
                    NULLIF(COUNT(CASE WHEN result IN ('W', 'L') THEN 1 END), 0) as win_pct
            FROM team_game_log
            WHERE year = %s AND team_id = %s
        )  
        SELECT 
            COALESCE(wins, 0) as wins,
            COALESCE(losses, 0) as losses,
            COALESCE(home_wins, 0) as home_wins,
            COALESCE(home_losses, 0) as home_losses,
            COALESCE(away_wins, 0) as away_wins,
            COALESCE(away_losses, 0) as away_losses,
            COALESCE(win_pct, 0) as win_pct
        FROM record_counts
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season, team_id))
            row = cur.fetchone()

            if row:
                return {
                    "wins": row[0],
                    "losses": row[1],
                    "home_wins": row[2],
                    "home_losses": row[3],
                    "away_wins": row[4],
                    "away_losses": row[5],
                    "win_pct": row[6],
                }
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while fetching teams home and away wins and losses for season {season} and team {team_id}: {e}"
        )
        raise e


def fetch_pks_for_inserted_team_game_logs(season: int):
    """
    Fetch previously inserted PKs for a specific season need to be updated

    Args:
        season (int): the season to fetch records for
    """

    sql = """
        SELECT team_id, week, year 
        FROM team_game_log 
        WHERE year = %s 
    """

    pks = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season,))
            rows = cur.fetchall()

            if rows:
                for row in rows:
                    if row:
                        pks.append({"team_id": row[0], "week": row[1], "year": row[2]})
            else:
                return []

        return pks

    except Exception as e:
        logging.error(
            f"An error occurred while fetching PKs for team_game_logs for season {season}: {e}"
        )
        raise e
