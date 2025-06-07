import logging
import pandas as pd
from scrape_and_score.db.connection import get_connection
import warnings
from scrape_and_score.constants import QUERY


def fetch_player_by_normalized_name(normalized_name: str):
    """
    Fetch player by normalized_name

    Args:
        normalized_name (str): the players name normalized
    """

    sql = "SELECT player_id, name, position, normalized_name, hashed_name FROM player WHERE normalized_name = %s"
    player = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (normalized_name,))
            row = cur.fetchone()

            if row:
                player = {
                    "player_id": row[0],
                    "name": row[1],
                    "position": row[2],
                    "normalized_name": row[3],
                    "hashed_name": row[4],
                }

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player with normalized name {normalized_name}: {e}."
        )
        raise e

    return player


def fetch_player_id_by_normalized_name_season_and_position(
    normalized_name: str, position: str, season: int
):
    """
    Fetch player ID corresponding to normalized name, position, and season

    Args:
        normalized_name (str): the name to extract player ID for
        position (str): the position extract player ID for
        season (int): relevant season

    Returns
        int: the player ID corresponding to args
    """

    sql = """
        SELECT p.player_id FROM player p
        JOIN player_teams pt ON p.player_id = pt.player_id 
        WHERE normalized_name LIKE %s AND pt.season = %s AND p.position = %s
    """
    player_id = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            player_name = normalized_name.split(" ")
            if len(player_name) > 2:
                player_last_name = player_name[1]
            else:
                player_last_name = player_name[-1]

            like_pattern = f"% {player_last_name}"
            cur.execute(sql, (like_pattern, season, position))
            row = cur.fetchone()

            if row:
                player_id = row[0]
            else:
                logging.warning(f"Unable to find player ID for {normalized_name}")

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player ID for {normalized_name}: {e}"
        )
        raise e

    return player_id


def fetch_player_seasonal_metrics(season: int):
    """
    Retrieve player seasonal metrics for a particular season

    Args:
        season (int): relevant season
    """

    sql = """
        SELECT 
            *
        FROM player_seasonal_scoring_metrics pssm
        LEFT JOIN player_seasonal_passing_metrics pspm ON pssm.player_id = pspm.player_id AND pssm.season = pspm.season 
        LEFT JOIN player_seasonal_rushing_receiving_metrics psrrm ON pssm.player_id = psrrm.player_id AND pssm.season = psrrm.season 
        WHERE pssm.season = %s
    """

    metrics = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season,))
            row = cur.fetchone()

            if row:
                colnames = [desc[0] for desc in cur.description]
                metrics = dict(zip(colnames, row))
            else:
                logging.warning(
                    f"No seasonal metrics persisted for players for season={season}"
                )

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player season metrics; season={season}: {e}"
        )
        raise e

    return metrics


def fetch_player_id_by_normalized_name(normalized_name: str):
    """
    Retrieve players ID by their normalized name

    Args:
        normalized_name (str): normalized name to retrieve player ID by

    Returns:
        player_id (int): the respective player ID (or None if not found)
    """

    sql = """
        SELECT player_id 
        FROM player 
        WHERE normalized_name LIKE %s
    """
    player_id = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            like_pattern = f"%{normalized_name}%"
            cur.execute(sql, (like_pattern,))  # use cleaned name
            row = cur.fetchone()

            if row:
                player_id = row[0]
            else:
                logging.warning(f"Unable to find player ID for {normalized_name}")

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player ID for {normalized_name}: {e}"
        )
        raise e

    return player_id


def fetch_player_game_log_by_pk(pk: dict):
    """
    Functionality to retrieve a players game log by its PK (player_id, week, year)

    Args:
        pk (dict): primary key for a given player's game log (player_id, week, year)
    Returns:
        player_game_log (dict): the player game log corresponding to the given PK
    """

    sql = "SELECT * FROM player_game_log WHERE player_id=%s AND week=%s AND year=%s"
    player_game_log = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (pk["player_id"], pk["week"], pk["year"]))
            row = cur.fetchone()

            if row:
                player_game_log = {
                    "player_id": row[0],
                    "week": row[1],
                    "day": row[2],
                    "year": row[3],
                    "home_team": row[4],
                    "opp": row[5],
                    "result": row[6],
                    "points_for": row[7],
                    "points_allowed": row[8],
                    "completions": row[9],
                    "attempts": row[10],
                    "pass_yd": row[11],
                    "pass_td": row[12],
                    "interceptions": row[13],
                    "rating": row[14],
                    "sacked": row[15],
                    "rush_att": row[16],
                    "rush_yds": row[17],
                    "rush_tds": row[18],
                    "tgt": row[19],
                    "rec": row[20],
                    "rec_yd": row[21],
                    "rec_td": row[22],
                    "snap_pct": row[23],
                    "fantasy_points": row[24],
                    "off_snps": row[25],
                }

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the player game log corresponding to PK {pk}: {e}"
        )
        raise e

    return player_game_log


def fetch_all_player_game_logs_for_recent_week(year: int):
    """
    Functionality to retrieve all player game logs for the most recent week

    Args:
        year (int): year to fetch game logs for

    Returns:
        game_logs (list): list of game logs for given year & recent week
    """

    sql = "SELECT * FROM player_game_log WHERE year=%s AND week=(SELECT MAX(week) FROM player_game_log)"
    player_game_logs = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year,))
            rows = cur.fetchall()

            if rows:
                for row in rows:
                    player_game_log = {
                        "player_id": row[0],
                        "week": row[1],
                        "day": row[2],
                        "year": row[3],
                        "home_team": row[4],
                        "opp": row[5],
                        "result": row[6],
                        "points_for": row[7],
                        "points_allowed": row[8],
                        "completions": row[9],
                        "attempts": row[10],
                        "pass_yd": row[11],
                        "pass_td": row[12],
                        "interceptions": row[13],
                        "rating": row[14],
                        "sacked": row[15],
                        "rush_att": row[16],
                        "rush_yds": row[17],
                        "rush_tds": row[18],
                        "tgt": row[19],
                        "rec": row[20],
                        "rec_yd": row[21],
                        "rec_td": row[22],
                        "snap_pct": row[23],
                    }
                    player_game_logs.append(player_game_log)

    except Exception as e:
        logging.error(
            f"An error occurred while fetching all recent week player game logs: {e}"
        )
        raise e

    return player_game_logs


def fetch_all_player_game_logs_for_given_year(year: int):
    """
    Functionality to retrieve all player game logs for a given year

    Args:
        year (int): year to fetch game logs for

    Returns:
        game_logs (list): list of player game logs for given year
    """

    sql = "SELECT * FROM player_game_log WHERE year=%s"
    player_game_logs = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year,))
            rows = cur.fetchall()

            if rows:
                for row in rows:
                    player_game_log = {
                        "player_id": row[0],
                        "week": row[1],
                        "day": row[2],
                        "year": row[3],
                        "home_team": row[4],
                        "opp": row[5],
                        "result": row[6],
                        "points_for": row[7],
                        "points_allowed": row[8],
                        "completions": row[9],
                        "attempts": row[10],
                        "pass_yd": row[11],
                        "pass_td": row[12],
                        "interceptions": row[13],
                        "rating": row[14],
                        "sacked": row[15],
                        "rush_att": row[16],
                        "rush_yds": row[17],
                        "rush_tds": row[18],
                        "tgt": row[19],
                        "rec": row[20],
                        "rec_yd": row[21],
                        "rec_td": row[22],
                        "snap_pct": row[23],
                    }
                    player_game_logs.append(player_game_log)

    except Exception as e:
        logging.error(
            f"An error occurred while fetching all recent week player game logs: {e}"
        )
        raise e

    return player_game_logs


def fetch_player_betting_odds_record_by_pk(
    player_id: int, week: int, season: int, label: str
):
    """
    Retrieve a player betting odds record by PK

    Args:
        player_id (int): relevant player ID
        week (int): week to retrieve record for
        season (int): season to retrieve record for
        label (str): label to account for

    Args:
        dict: persisted record
    """

    sql = "SELECT player_id, label, cost, line, week, season FROM player_betting_odds WHERE player_id = %s AND week= %s AND season = %s AND label = %s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_id, week, season, label))
            row = cur.fetchone()

            if row:
                return {
                    "player_id": row[0],
                    "label": row[1],
                    "cost": row[2],
                    "line": row[3],
                    "week": row[4],
                    "season": row[5],
                }
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the team game logs corresponding to team {team_id} and year {year}: {e}"
        )
        raise e


def fetch_players_active_in_specified_year(year):
    """
    Retrieve all relevant player names & ids that are active in a specified year

    Args:
        year (int): season to retrieve data for

    Return
        players_names (list): all relevant player names
    """

    sql = """
      SELECT 
	    DISTINCT p.name,
        p.player_id
      FROM 
	    player p
      JOIN player_game_log pgl 
	    ON pgl.player_id = p.player_id
      WHERE 
	    pgl.year = %s
    """

    names = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year,))
            rows = cur.fetchall()

            for row in rows:
                names.append({"name": row[0], "id": row[1]})

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the relevant player names & ids corresponding to year {year}: {e}"
        )
        raise e

    return names


def fetch_players_on_a_roster_in_specific_year(year: int):
    """
    Retrieve players on an active roster in specific year

    Args:
        year (int): season to account for

    Returns:
        list: list of disitinct players
    """

    sql = """ 
        SELECT DISTINCT
            p.player_id,
            p.name,
            p.position,
            p.normalized_name,
            p.hashed_name,
            p.pfr_available
        FROM 
            player p 
        JOIN 
            player_teams pt 
        ON
            p.player_id = pt.player_id 
        WHERE 
            pt.season = %s
    """

    players = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year,))
            rows = cur.fetchall()

            for row in rows:
                players.append(
                    {
                        "player_id": row[0],
                        "player_name": row[1],
                        "position": row[2],
                        "normalized_name": row[3],
                        "hashed_name": row[4],
                        "pfr_available": row[5],
                    }
                )

    except Exception as e:
        logging.error(
            f"An error occurred while fetching players on active roster in season {year}: {e}"
        )
        raise e

    return players


def fetch_player_ids_of_players_who_have_advanced_metrics_persisted(
    year: int, week: int = None
):
    """
    Retrieve player IDs of players who have advanced metrics persisted

    Args:
        year (int): season to account for

    Returns:
        dict: set of disitinct players ids that are already persisted
    """

    if week is None:
        sql = """ 
            SELECT DISTINCT player_id
            FROM player_advanced_passing
            WHERE season = %s
            UNION
            SELECT DISTINCT player_id
            FROM player_advanced_rushing_receiving
            WHERE season = %s;
        """
    else:
        sql = """ 
            SELECT DISTINCT player_id
            FROM player_advanced_passing
            WHERE season = %s AND week = %s
            UNION
            SELECT DISTINCT player_id
            FROM player_advanced_rushing_receiving
            WHERE season = %s AND week = %s;
        """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year, year))
            rows = cur.fetchall()

            player_ids = {row[0] for row in rows}

            logging.info(
                f"Successfully fetched {len(player_ids)} player IDs of players in advanced rushing/receiving/passing tables for season {year} ."
            )

    except Exception as e:
        logging.error(
            f"An error occurred while fetching players ID of players in advanced rushing/receiving/passing tables for season {year}"
        )
        raise e

    return player_ids


def fetch_players_on_a_roster_in_specific_year_with_hashed_name(year: int):
    """
    Retrieve players on an active roster in specific year that have a hashed name persisted

    Args:
        year (int): season to account for

    Returns:
        list: list of disitinct players
    """

    sql = """ 
        SELECT DISTINCT
            p.player_id,
            p.name,
            p.position,
            p.normalized_name,
            p.hashed_name,
            p.pfr_available
        FROM 
            player p 
        JOIN 
            player_teams pt 
        ON
            p.player_id = pt.player_id 
        WHERE 
            pt.season = %s AND hashed_name IS NOT NULL
    """

    players = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year,))
            rows = cur.fetchall()

            for row in rows:
                players.append(
                    {
                        "player_id": row[0],
                        "player_name": row[1],
                        "position": row[2],
                        "normalized_name": row[3],
                        "hashed_name": row[4],
                        "pfr_available": row[5],
                    }
                )

            logging.info(
                f"Successfully fetched {len(players)} player records active in {year} with a hashed name persisted."
            )

    except Exception as e:
        logging.error(
            f"An error occurred while fetching players on active roster with hashed name persisted in season {year}: {e}"
        )
        raise e

    return players


def fetch_player_teams_record_by_pk(record: dict):
    """
    Retrieve player_teams record by PK

    Args:
        record (dict): mapping of PK's

    Returns:
        record (dict): record in DB
    """

    sql = "SELECT * FROM player_teams WHERE player_id = %s AND team_id = %s AND season = %s AND strt_wk = %s AND end_wk = %s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(
                sql,
                (
                    record["player_id"],
                    record["team_id"],
                    record["season"],
                    record["strt_wk"],
                    record["end_wk"],
                ),
            )
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


def fetch_player_teams_records_by_player_and_season(player_id: int, season: int):
    """
    Retrieve player_teams records by player ID / season

    Args:
        player_id (int): the players ID
        season (int): relevant season

    Returns:
        record (dict): record in DB
    """

    sql = "SELECT player_id, team_id, season, strt_wk, end_wk FROM player_teams WHERE player_id = %s AND season = %s"
    records = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_id, season))
            rows = cur.fetchall()

            for row in rows:
                records.append(
                    {
                        "player_id": row[0],
                        "team_id": row[1],
                        "season": row[2],
                        "strt_wk": row[3],
                        "end_wk": row[4],
                    }
                )

        return records

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving player_teams record for player_id {player_id} and season {season}: {e}"
        )
        raise e


def fetch_player_depth_chart_position_by_pk(record: dict):
    """
    Retrieve player_depth_chart record by PK

    Args:
        record (dict): mapping of PK's

    Returns:
        int: depth chart position
    """

    sql = "SELECT depth_chart_pos FROM player_depth_chart WHERE player_id = %s AND season = %s AND week = %s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (record["player_id"], record["season"], record["week"]))
            row = cur.fetchone()

            if row:
                return int(row[0])
            else:
                return None

    except Exception as e:
        logging.error(f"An error occurred while retrieving player depth chart: {e}")
        raise e


def get_count_player_demographics_records_for_season(season: int):
    """Functionality to retrieve the count of player_demographcis records for a particular season

    Args:
        season (int): the season pertaining to the record

    Returns:
        int: number of player demographic records for season
    """

    sql = "SELECT * FROM player_demographics WHERE season = %s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season,))
            rows = cur.fetchall()

            return len(rows)

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving the count of player_demographic records pertaining to season {season}",
            exc_info=True,
        )
        raise e


def get_count_player_teams_records_for_season(season: int):
    """Functionality to retrieve the count of player_teams records for a particular season

    Args:
        season (int): the season pertaining to the record

    Returns:
        int: number of player_teams records for season
    """

    sql = "SELECT * FROM player_teams WHERE season = %s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season,))
            rows = cur.fetchall()

            return len(rows)

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving the count of player_teams records pertaining to season {season}",
            exc_info=True,
        )
        raise e


def fetch_player_teams_by_week_season_and_player_id(
    season: int, week: int, player_id: int
):
    """
    Retrieve playe teams record by week/season/player_id

    Args:
        season (int): relevant season
        week (int): relevant week
        player_id (int): relevant player ID
    """

    sql = "SELECT team_id FROM player_teams WHERE season = %s AND %s >= strt_wk AND %s <= end_wk AND player_id = %s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season, week, week, player_id))
            row = cur.fetchone()

            if not row:
                logging.warning(
                    f"No players found corresponding to week {week} and player ID {player_id} of the {season} NFL season"
                )
                return None
            else:
                return row[0]

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving players team corresponding to season {season}, week {week}, and player ID {player_id}",
            exc_info=True,
        )
        raise e

    return players


def fetch_players_corresponding_to_season_week_team(
    season: int, week: int, team_id: int
):
    """
    Fetch player_ids corresponding to particular season, week, and team_id

    Args:
        season (int): relevant NFL season
        week (int): relevant week
        team_id (int): relevant teams

    Returns:
        list: player_ids corresponding to this week/season/team
    """

    sql = "SELECT player_id FROM player_teams WHERE season = %s AND %s >= strt_wk AND %s <= end_wk AND team_id = %s"
    players = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season, week, week, team_id))
            rows = cur.fetchall()

            if not rows:
                logging.warning(
                    f"No players found corresponding to week {week} and team ID {team_id} of the {season} NFL season"
                )
                return players

            for row in rows:
                players.append(row[0])

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving players corresponding to season {season}, week {week}, and team_id {team_id}",
            exc_info=True,
        )
        raise e

    return players


def retrieve_player_demographics_record_by_pk(season: int, player_id: int):
    """Functionality to retrieve a player demographic record by its PK

    Args:
        season (int): the season pertaining to the record
        player_id (int): the player_id pertaining to the record
    """

    sql = "SELECT * FROM player_demographics WHERE player_id = %s AND season = %s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_id, season))
            row = cur.fetchone()

            if row:
                return row[0]
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving player_demographic record pertaining to season {season} and player_id {player_id}",
            exc_info=True,
        )
        raise e


def fetch_pks_for_inserted_player_injury_records(season: int, week: int):
    """
    Fetch previously inserted PKs for a specific season / week in order to determine which records need to be updated

    Args:
        season (int): the season to fetch records for
        week (int): the week to fetch records for
    """

    sql = """
        SELECT player_id, week, season 
        FROM player_injuries 
        WHERE season = %s AND week = %s
    """

    pks = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (season, week))
            rows = cur.fetchall()

            if rows:
                for row in rows:
                    if row:
                        pks.append(
                            {"player_id": row[0], "week": row[1], "season": row[2]}
                        )
            else:
                return []

        return pks

    except Exception as e:
        logging.error(
            f"An error occurred while fetching PKs for player_injuries for season {season} and week {week}: {e}"
        )
        raise e


def fetch_player_demographic_record(player_id: int, season: int):
    """
    Fetch player_demographics record from db

    Args:
        player_id (int): the player we want to retreive the record for
    """

    sql = """
        SELECT player_id, season, age, height, weight
        FROM player_demographics
        WHERE player_id = %s AND season = %s
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_id, season))
            row = cur.fetchone()

            if row:
                return {
                    "player_id": row[0],
                    "season": row[1],
                    "age": row[2],
                    "height": row[3],
                    "weight": row[4],
                }
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player_demographic records for player {player_id} of the {season} NFL season: {e}"
        )
        raise e


def fetch_player_date_of_birth(player_id: int):
    """
    Fetch players date of birth

    Args:
        player_id (int): the player we want to retrieve date of birth for
    """

    sql = """
        SELECT dob
        FROM player
        WHERE player_id = %s
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_id,))
            row = cur.fetchone()

            if row:
                return row[0]
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player date of birth corresponding to player ID {player_id}: {e}"
        )
        raise e


def fetch_player_name_by_id(player_id: int):
    """
    Fetch players name by their ID

    Args:
        player_id (int): the player we want to retrieve name for
    """

    sql = """
        SELECT name
        FROM player
        WHERE player_id = %s
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_id,))
            row = cur.fetchone()

            if row:
                return row[0]
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player name corresponding to player ID {player_id}: {e}"
        )
        raise e


def fetch_independent_and_dependent_variables(week: int, season: int):
    """
    Functionality to retrieve the needed dependent and independent variables needed
    to create our predicition models

    Args:
        week (int): either the relevant week or None
        season (int): the relevant season or None

    Returns:
    df (pd.DataFrame): data frame containing results of query

    """

    # generate where clause in the case this is for predictions
    where_clause = (
        f" WHERE pgl.week = {week} AND pgl.year = {season}"
        if week is not None and season is not None
        else ""
    )

    sql = QUERY + where_clause
    df = None

    try:
        connection = get_connection()

        # filter warnings regarding using pyscopg2 connection
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df = pd.read_sql_query(sql, connection)

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the data needed to create mutliple linear regression model: {e}"
        )
        raise e

    return df