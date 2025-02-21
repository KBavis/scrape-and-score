import logging
import pandas as pd
from .connection import get_connection
import warnings

"""
Functionality to fetch multiple teams 
"""


def fetch_all_teams():
    sql = "SELECT * FROM team"
    teams = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

            for row in rows:
                teams.append(
                    {
                        "team_id": row[0],
                        "name": row[1],
                        "offense_rank": row[2],
                        "defense_rank": row[3],
                    }
                )

    except Exception as e:
        logging.error("An error occurred while fetching all teams: {e}")
        raise e

    return teams


"""
Functionality to fetch a team by their team name 

Args:
   team_name (str): team name to retrieve team by

Returns:
   team (dict): team record
"""


def fetch_team_by_name(team_name: int):
    sql = "SELECT * FROM team WHERE name = %s"
    team = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (team_name,))  # ensure team_name in tuple
            row = cur.fetchone()

            if row:
                team = {
                    "team_id": row[0],
                    "name": row[1],
                    "offense_rank": row[2],
                    "defense_rank": row[3],
                }

    except Exception as e:
        logging.error(
            f"An error occurred while fetching team with name {team_name}: {e}"
        )
        raise e

    return team


"""
Functionality to fetch all players

Returns:
   players (list): list of players persisted in DB
"""


def fetch_all_players():
    sql = "SELECT * FROM player"
    players = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

            for row in rows:
                players.append(
                    {
                        "player_id": row[0],
                        "team_id": row[1],
                        "player_name": row[2],
                        "position": row[3],
                    }
                )

    except Exception as e:
        logging.error(f"An error occurred while fetching all players: {e}")
        raise e

    return players


"""
Functionality to fetch a player by their player name 

Args:
   player_name (str): player name to retrieve player by

Returns:
   player (dict): player record
"""


def fetch_player_by_name(player_name: str):
    sql = "SELECT * FROM player WHERE name = %s"
    player = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_name,))  # ensure player_name is passed as a tuple
            row = cur.fetchone()

            if row:
                player = {
                    "player_id": row[0],
                    "team_id": row[1],
                    "name": row[2],
                    "position": row[3],
                }

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player with name {player_name}: {e}."
        )
        raise e

    return player


"""
Functionality to retrieve a single team game log from our DB 

Args:
   None 
Returns: 
   player_game_log (dict): team game log or None if not found 
"""


def fetch_one_team_game_log():
    sql = "SELECT * FROM team_game_log FETCH FIRST 1 ROW ONLY"
    team_game_log = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()

            if row:
                team_game_log = {"team_id": row[0], "week": row[1], "year": row[3]}

    except Exception as e:
        logging.error(
            f"An error occurred while fetching one record from team_game_log: {e}"
        )
        raise e

    return team_game_log


"""
Functionality to retrieve a single player game log from our DB 

Args:
   None 
Returns: 
   player_game_log (dict): player game log or None if not found 
"""


def fetch_one_player_game_log():
    sql = "SELECT * FROM player_game_log FETCH FIRST 1 ROW ONLY"
    player_game_log = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()

            if row:
                player_game_log = {"player_id": row[0], "week": row[1], "year": row[3]}

    except Exception as e:
        logging.error(
            f"An error occurred while fetching one record from player_game_log: {e}"
        )
        raise e

    return player_game_log


"""
Functionality to retrieve a team's game log by its PK (team_id, week, year)

Args:
   pk (dict): primary key for a given team's game log (team_id, week, year)

   Returns:
   team_game_log (dict): the team game log corresponding to the given PK 
"""


def fetch_team_game_log_by_pk(pk: dict):
    sql = "SELECT * FROM team_game_log WHERE team_id=%s AND week=%s AND year=%s"
    team_game_log = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (pk["team_id"], pk["week"], pk["year"]))
            row = cur.fetchone()

            if row:
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

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the team game log corresponding to PK {pk}: {e}"
        )
        raise e

    return team_game_log


"""
Functionality to retrieve a players game log by its PK (player_id, week, year)

Args:
   pk (dict): primary key for a given player's game log (player_id, week, year)
Returns:
   player_game_log (dict): the player game log corresponding to the given PK 
"""


def fetch_player_game_log_by_pk(pk: dict):
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
                }

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the player game log corresponding to PK {pk}: {e}"
        )
        raise e

    return player_game_log


"""
Functionality to retrieve all player game logs for the most recent week 

Args:
   year (int): year to fetch game logs for 

Returns:
   game_logs (list): list of game logs for given year & recent week
"""


def fetch_all_player_game_logs_for_recent_week(year: int):
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


"""
Functionality to retrieve all player game logs for a given year 

Args:
   year (int): year to fetch game logs for 
   
Returns:
   game_logs (list): list of player game logs for given year
"""


def fetch_all_player_game_logs_for_given_year(year: int):
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


"""
Functionality to retrieve teams game logs for a particular season 

Args:
   year (int): year to fetch team game logs for 
   team_id (int): team to fetch game logs for 

Returns
   game_logs (list): list of game_logs for a particular season/team
"""


def fetch_all_teams_game_logs_for_season(team_id: int, year: int):
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


"""
Functionality to retrieve the needed dependent and independent variables needed 
to create our multiple linear regression based model 

TODO (FFM-145): Include external factors such as weather, vegas projections, players depth chart position 

Args:
   None 

Returns:
   df (pd.DataFrame): data frame containing results of query 

"""


def fetch_independent_and_dependent_variables_for_mult_lin_regression():
    sql = """
    WITH PlayerProps AS (
        SELECT 
            pbo.player_name,
            pbo.week,
            pbo.season,
            jsonb_agg(
                json_build_object(
                    'label', pbo.label,
                    'line', pbo.line,
                    'cost', pbo.cost
                )
            ) AS props
        FROM 
            player_betting_odds pbo
        GROUP BY
            pbo.player_name, pbo.week, pbo.season
    )
	  SELECT
         p.player_id,
         p.position,
         pgl.fantasy_points,
         t.off_rush_rank,
         t.off_pass_rank,
         td.def_rush_rank,
         td.def_pass_rank,
         tbo.game_over_under,
         tbo.spread,
         CASE
            WHEN tbo.favorite_team_id = t.team_id THEN 1
			ELSE 0
         END AS is_favorited,
		 pp.props
      FROM
         player_game_log pgl
      JOIN 
         player p ON p.player_id = pgl.player_id 
      JOIN 
         team t ON p.team_id = t.team_id
      JOIN 
         team td ON pgl.opp = td.team_id
      JOIN
	  	 PlayerProps pp ON p.name = pp.player_name AND pgl.week = pp.week AND pgl.year = pp.season
	  JOIN 
         team_betting_odds tbo 
      ON 
				(tbo.home_team_id = t.team_id OR tbo.home_team_id = td.team_id)
               AND 
				(tbo.away_team_id = t.team_id OR tbo.away_team_id = td.team_id);	
   """

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


"""
Retrieve relevant inputs for a player in order to make prediction

TODO: Update this to account for player props 

Args:
   week (int): the week corresponding to the matchup we are predicitng for
   season (int): the season to fetch data from
   player_name (str): name of the player we want to predict fantasy points for 

Returns:
   df (pd.DataFrame): data frame containing relevant inputs
"""


def fetch_inputs_for_prediction(week: int, season: int, player_name: str):
    sql = """
    WITH PlayerProps AS (
        SELECT 
            pbo.player_name,
            pbo.week,
            pbo.season,
            jsonb_agg(
                json_build_object(
                    'label', pbo.label,
                    'line', pbo.line,
                    'cost', pbo.cost
                )
            ) AS props
        FROM 
            player_betting_odds pbo
        WHERE
            pbo.player_name = %s
        AND
            pbo.week = %s
        AND 
            pbo.season = %s
        GROUP BY
            pbo.player_name, pbo.week, pbo.season
    )
    SELECT
        p.position,
        ROUND(CAST(player_avg.avg_fantasy_points AS NUMERIC), 2) AS avg_fantasy_points,
        t.off_rush_rank,
        t.off_pass_rank,
        df.def_rush_rank,
        df.def_pass_rank,
        tbo.game_over_under,
        tbo.spread,
        CASE 
            WHEN tbo.favorite_team_id = t.team_id THEN 1
            ELSE 0
        END AS is_favorited,
        pp.props
    FROM
        player_game_log pgl
    JOIN 
        player p ON p.player_id = pgl.player_id 
    JOIN 
        team t ON p.team_id = t.team_id
    JOIN 
        team df ON df.team_id = pgl.opp 
    JOIN
        team_betting_odds tbo 
    ON ((tbo.home_team_id = t.team_id OR tbo.away_team_id = t.team_id) 
       AND (tbo.home_team_id = df.team_id OR tbo.away_team_id = df.team_id))
    JOIN (
        SELECT 
            pgl.player_id, 
            AVG(pgl.fantasy_points) AS avg_fantasy_points
            FROM player_game_log pgl
            WHERE pgl.player_id = (SELECT player_id FROM player WHERE name = %s)
            AND pgl.year = %s 
            GROUP BY pgl.player_id
    ) player_avg 
    ON 
        player_avg.player_id = pgl.player_id
    JOIN PlayerProps pp 
        ON pp.player_name = p.name AND pp.week = tbo.week AND pp.season = tbo.season
    WHERE 
        p.name = %s AND
        tbo.week = %s AND tbo.season = %s
    GROUP BY 
        p.position, t.off_rush_rank, t.off_pass_rank, df.def_rush_rank, df.def_pass_rank, 
        tbo.game_over_under, tbo.spread, tbo.favorite_team_id, player_avg.avg_fantasy_points, is_favorited, pp.props;
    """

    df = None

    try:
        connection = get_connection()

        # filter warnings regarding using pyscopg2 connection
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df = pd.read_sql_query(
                sql, connection, params=(player_name, week, season, player_name, season, player_name, week, season)
            )

            if df.empty:
                raise Exception(
                    f"Unable to extract relevant prediction inputs for player {player_name} for the week/season {week} {season}"
                )

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the data needed to make prediction via our linear regression model: {e}"
        )
        raise e

    return df

""" 
Retrieve all relevant player names & ids that are active in a specified year 

Args:
    year (int): season to retrieve data for 

Return
    players_names (list): all relevant player names 
"""
def fetch_players_active_in_specified_year(year):
    sql = '''
      SELECT 
	    DISTINCT p.name,
        p.player_id
      FROM 
	    player p
      JOIN player_game_log pgl 
	    ON pgl.player_id = p.player_id
      WHERE 
	    pgl.year = %s
    '''
    
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
    
    


"""
Retrieve the latest week we have persisted data in our 'team_betting_odds' table 

Args:
   year (int): season to retrieve data for 

Return:
   week (int): the latest week persisted in our db table
"""


def fetch_max_week_persisted_in_team_betting_odds_table(year: int):
    sql = "SELECT week FROM team_betting_odds WHERE season = %s AND week = (SELECT MAX(week) FROM team_betting_odds)"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year,))
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


"""
Retrieve favorite_team_id, spread, and game_over_under from team_betting_odds table if a corresponding entry exists 

Args:
   player_name (str): players name to check for 
   week (int): week to check for 
   year (int): year to fetch data for 

Returns:
   valid (bool): flag indicating if record existed 
"""


def validate_week_and_corresponding_player_entry_exists(
    player_name: str, week: int, year: int
):
    sql = """
      SELECT 
         favorite_team_id, 
         spread, 
         game_over_under 
      FROM 
         team_betting_odds 
      WHERE 
         ((home_team_id = (SELECT team_id FROM player WHERE name = %s))
            OR 
         (away_team_id = (SELECT team_id FROM player WHERE name = %s)))
         AND 
         week = %s AND season = %s 
   """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_name, player_name, week, year))
            row = cur.fetchone()

            if row:
                return True
            else:
                return False

    except Exception as e:
        logging.error(
            f"An error occurred while fetching the latest week persisted in team_betting_odds for year {year}: {e}"
        )
        raise e
