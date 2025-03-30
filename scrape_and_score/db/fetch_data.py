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
                        "name": row[1]
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
                    "name": row[1]
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
    sql = "SELECT player_id, player_name, position, normalized_name, hashed_name FROM player"
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
                        "player_name": row[1],
                        "position": row[2],
                        "normalized_name": row[3],
                        "hashed_name": row[4]
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
    sql = "SELECT player_id, name, position, normalized_name, hashed_name FROM player WHERE name = %s"
    player = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_name,))  # ensure player_name is passed as a tuple
            row = cur.fetchone()

            if row:
                player = {
                    "player_id": row[0],
                    "name": row[1],
                    "position": row[2],
                    "normalized_name": row[3],
                    "hashed_name": row[4]
                }

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player with name {player_name}: {e}."
        )
        raise e

    return player



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
                    "hashed_name": row[4]
                }

    except Exception as e:
        logging.error(
            f"An error occurred while fetching player with normalized name {normalized_name}: {e}."
        )
        raise e

    return player



def fetch_player_id_by_normalized_name(normalized_name: str):

    sql = """
        SELECT player_id 
        FROM player 
        WHERE normalized_name = %s
    """
    player_id = None

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (normalized_name,))  # use cleaned name
            row = cur.fetchone()

            if row:
                player_id = row[0]
            else:
                logging.warning(f"Unable to find player ID for {normalized_name}")

    except Exception as e:
        logging.error(f"An error occurred while fetching player ID for {normalized_name}: {e}")
        raise e

    return player_id


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
to create our predicition models 

TODO (FFM-145): Include external factors such as weather, vegas projections, players depth chart position 

Args:
   None 

Returns:
   df (pd.DataFrame): data frame containing results of query 

"""

#TODO: Optimize query so it doesn't take forever to run 
#TODO: Ensure that the we update join statement for previous year data to ensure we don't just exclude all rookies!
def fetch_independent_and_dependent_variables():
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
	  	 -- player position & ID 
         p.player_id,
         p.position,
		 -- player total fantasy points scored 
         pgl.fantasy_points,
		 -- aggregate metrics 
		 pam.avg_fantasy_points AS avg_wkly_fantasy_points,
		 --depth chart position 
		 pdc.depth_chart_pos AS depth_chart_position,
		 -- game date/year
		 pgl.week,
         pgl.year as season,
		 -- weekly rankings throughout season
         t_tr.off_rush_rank,
         t_tr.off_pass_rank,
         t_td.def_rush_rank,
         t_td.def_pass_rank,
		 -- team betting odds 
         tbo.game_over_under,
         tbo.spread,
		 -- player demographics 
		 pd.age,
		 pd.height,
		 pd.weight,
		 -- players team previous year general stats
		 tsgm.fumble_lost as prev_year_team_total_fumbles_lost,
		 tsgm.home_wins as prev_year_team_totals_home_wins,
		 tsgm.home_losses as prev_year_team_total_home_losses,
		 tsgm.away_wins as prev_year_team_total_away_wins,
		 tsgm.away_losses as prev_year_team_total_away_losses,
		 tsgm.wins as prev_year_team_total_wins,
		 tsgm.losses as prev_year_team_total_losses, 
		 tsgm.win_pct as prev_year_team_total_win_pct,
		 tsgm.total_games as prev_year_team_total_games,
		 tsgm.total_yards as prev_year_team_total_yards,
		 tsgm.plays_offense as prev_year_team_total_plays_offense,
		 tsgm.yds_per_play as prev_year_team_yds_per_play,
		 tsgm.turnovers as prev_year_team_total_turnovers,
		 tsgm.first_down as prev_year_team_total_first_downs, 
		 tsgm.penalties as prev_year_team_total_penalties,
         tsgm.penalties_yds as prev_year_team_total_penalties_yds,
         tsgm.pen_fd as prev_year_team_total_pen_fd,
         tsgm.drives as prev_year_team_total_drives,
         tsgm.score_pct as prev_year_team_total_score_pct,
         tsgm.turnover_pct as prev_year_team_total_turnover_pct,
         tsgm.start_avg as prev_year_team_total_start_avg,
         tsgm.time_avg as prev_year_team_total_time_avg,
         tsgm.plays_per_drive as prev_year_team_total_plays_per_drive,
         tsgm.yds_per_drive as prev_year_team_total_yds_per_drive,
         tsgm.points_avg as prev_year_team_total_points_avg,
         tsgm.third_down_att as prev_year_team_total_third_down_att,
         tsgm.third_down_success as prev_year_team_total_third_down_success,
         tsgm.third_down_pct as prev_year_team_total_third_down_pct,
         tsgm.fourth_down_att as prev_year_team_total_fourth_down_att,
         tsgm.fourth_down_success as prev_year_team_total_fourth_down_success,
         tsgm.fourth_down_pct as prev_year_team_total_fourth_down_pct,
         tsgm.red_zone_att as prev_year_team_total_red_zone_att,
         tsgm.red_zone_scores as prev_year_team_total_red_zone_scores,
         tsgm.red_zone_pct as prev_year_team_total_red_zone_pct,
         -- opposing team previous year general stats
         opp_tsgm.fumble_lost as prev_year_opp_total_fumbles_lost,
         opp_tsgm.home_wins as prev_year_opp_totals_home_wins,
         opp_tsgm.home_losses as prev_year_opp_total_home_losses,
         opp_tsgm.away_wins as prev_year_opp_total_away_wins,
         opp_tsgm.away_losses as prev_year_opp_total_away_losses,
         opp_tsgm.wins as prev_year_opp_total_wins,
         opp_tsgm.losses as prev_year_opp_total_losses,
         opp_tsgm.win_pct as prev_year_opp_total_win_pct,
         opp_tsgm.total_games as prev_year_opp_total_games,
         opp_tsgm.total_yards as prev_year_opp_total_yards,
         opp_tsgm.plays_offense as prev_year_opp_total_plays_offense,
         opp_tsgm.yds_per_play as prev_year_opp_yds_per_play,
         opp_tsgm.turnovers as prev_year_opp_total_turnovers,
         opp_tsgm.first_down as prev_year_opp_total_first_downs,
         opp_tsgm.penalties as prev_year_opp_total_penalties,
         opp_tsgm.penalties_yds as prev_year_opp_total_penalties_yds,
         opp_tsgm.pen_fd as prev_year_opp_total_pen_fd,
         opp_tsgm.drives as prev_year_opp_total_drives,
         opp_tsgm.score_pct as prev_year_opp_total_score_pct,
         opp_tsgm.turnover_pct as prev_year_opp_total_turnover_pct,
         opp_tsgm.start_avg as prev_year_opp_total_start_avg,
         opp_tsgm.time_avg as prev_year_opp_total_time_avg,
         opp_tsgm.plays_per_drive as prev_year_opp_total_plays_per_drive,
         opp_tsgm.yds_per_drive as prev_year_opp_total_yds_per_drive,
         opp_tsgm.points_avg as prev_year_opp_total_points_avg,
         opp_tsgm.third_down_att as prev_year_opp_total_third_down_att,
         opp_tsgm.third_down_success as prev_year_opp_total_third_down_success,
         opp_tsgm.third_down_pct as prev_year_opp_total_third_down_pct,
         opp_tsgm.fourth_down_att as prev_year_opp_total_fourth_down_att,
         opp_tsgm.fourth_down_success as prev_year_opp_total_fourth_down_success,
         opp_tsgm.fourth_down_pct as prev_year_opp_total_fourth_down_pct,
         opp_tsgm.red_zone_att as prev_year_opp_total_red_zone_att,
         opp_tsgm.red_zone_scores as prev_year_opp_total_red_zone_scores,
         opp_tsgm.red_zone_pct as prev_year_opp_total_red_zone_pct,
		 -- players team previous year passing stats 
		 tspassingmetrics.pass_attempts as prev_year_team_total_pass_attempts,
         tspassingmetrics.complete_pass as prev_year_team_total_complete_pass,
         tspassingmetrics.incomplete_pass as prev_year_team_total_incomplete_pass,
         tspassingmetrics.passing_yards as prev_year_team_total_passing_yards,
         tspassingmetrics.pass_td as prev_year_team_total_pass_td,
         tspassingmetrics.interception as prev_year_team_total_interception,
         tspassingmetrics.net_yds_per_att as prev_year_team_total_net_yds_per_att,
         tspassingmetrics.first_downs as prev_year_team_total_first_downs,
         tspassingmetrics.cmp_pct as prev_year_team_total_cmp_pct,
         tspassingmetrics.td_pct as prev_year_team_total_td_pct,
         tspassingmetrics.int_pct as prev_year_team_total_int_pct,
         tspassingmetrics.success as prev_year_team_total_success,
         tspassingmetrics.long as prev_year_team_total_long,
         tspassingmetrics.yds_per_att as prev_year_team_total_yds_per_att,
         tspassingmetrics.adj_yds_per_att as prev_year_team_total_adj_yds_per_att,
         tspassingmetrics.yds_per_cmp as prev_year_team_total_yds_per_cmp,
         tspassingmetrics.yds_per_g as prev_year_team_total_yds_per_g,
         tspassingmetrics.rating as prev_year_team_total_rating,
         tspassingmetrics.sacked as prev_year_team_total_sacked,
         tspassingmetrics.sacked_yds as prev_year_team_total_sacked_yds,
         tspassingmetrics.sacked_pct as prev_year_team_total_sacked_pct,
         tspassingmetrics.adj_net_yds_per_att as prev_year_team_total_adj_net_yds_per_att,
         tspassingmetrics.comebacks as prev_year_team_total_comebacks,
         tspassingmetrics.game_winning_drives as prev_year_team_total_game_winning_drives,
         -- opposing team previous year passing stats
         opp_tspassingmetrics.pass_attempts as prev_year_opp_total_pass_attempts,
         opp_tspassingmetrics.complete_pass as prev_year_opp_total_complete_pass,
         opp_tspassingmetrics.incomplete_pass as prev_year_opp_total_incomplete_pass,
         opp_tspassingmetrics.passing_yards as prev_year_opp_total_passing_yards,
         opp_tspassingmetrics.pass_td as prev_year_opp_total_pass_td,
         opp_tspassingmetrics.interception as prev_year_opp_total_interception,
         opp_tspassingmetrics.net_yds_per_att as prev_year_opp_total_net_yds_per_att,
         opp_tspassingmetrics.first_downs as prev_year_opp_total_first_downs,
         opp_tspassingmetrics.cmp_pct as prev_year_opp_total_cmp_pct,
         opp_tspassingmetrics.td_pct as prev_year_opp_total_td_pct,
         opp_tspassingmetrics.int_pct as prev_year_opp_total_int_pct,
         opp_tspassingmetrics.success as prev_year_opp_total_success,
         opp_tspassingmetrics.long as prev_year_opp_total_long,
         opp_tspassingmetrics.yds_per_att as prev_year_opp_total_yds_per_att,
         opp_tspassingmetrics.adj_yds_per_att as prev_year_opp_total_adj_yds_per_att,
         opp_tspassingmetrics.yds_per_cmp as prev_year_opp_total_yds_per_cmp,
         opp_tspassingmetrics.yds_per_g as prev_year_opp_total_yds_per_g,
         opp_tspassingmetrics.rating as prev_year_opp_total_rating,
         opp_tspassingmetrics.sacked as prev_year_opp_total_sacked,
         opp_tspassingmetrics.sacked_yds as prev_year_opp_total_sacked_yds,
         opp_tspassingmetrics.sacked_pct as prev_year_opp_total_sacked_pct,
         opp_tspassingmetrics.adj_net_yds_per_att as prev_year_opp_total_adj_net_yds_per_att,
         opp_tspassingmetrics.comebacks as prev_year_opp_total_comebacks,
         opp_tspassingmetrics.game_winning_drives as prev_year_opp_total_game_winning_drives,
		 -- players team previous year rushing/receiving stats
		 tsrm.rush_att as prev_year_team_total_rush_att,
         tsrm.rush_yds_per_att as prev_year_team_total_rush_yds_per_att,
         tsrm.rush_fd as prev_year_team_total_rush_fd,
         tsrm.rush_success as prev_year_team_total_rush_success,
         tsrm.rush_long as prev_year_team_total_rush_long,
         tsrm.rush_yds_per_g as prev_year_team_total_rush_yds_per_g,
         tsrm.rush_att_per_g as prev_year_team_total_rush_att_per_g,
         tsrm.rush_yds as prev_year_team_total_rush_yds,
         tsrm.rush_tds as prev_year_team_total_rush_tds,
         tsrm.targets as prev_year_team_total_targets,
         tsrm.rec as prev_year_team_total_rec,
         tsrm.rec_yds as prev_year_team_total_rec_yds,
         tsrm.rec_yds_per_rec as prev_year_team_total_rec_yds_per_rec,
         tsrm.rec_td as prev_year_team_total_rec_td,
         tsrm.rec_first_down as prev_year_team_total_rec_first_down,
         tsrm.rec_success as prev_year_team_total_rec_success,
         tsrm.rec_long as prev_year_team_total_rec_long,
         tsrm.rec_per_g as prev_year_team_total_rec_per_g,
         tsrm.rec_yds_per_g as prev_year_team_total_rec_yds_per_g,
         tsrm.catch_pct as prev_year_team_total_catch_pct,
         tsrm.rec_yds_per_tgt as prev_year_team_total_rec_yds_per_tgt,
         tsrm.touches as prev_year_team_total_touches,
         tsrm.yds_per_touch as prev_year_team_total_yds_per_touch,
         tsrm.yds_from_scrimmage as prev_year_team_total_yds_from_scrimmage,
         tsrm.rush_receive_td as prev_year_team_total_rush_receive_td,
         tsrm.fumbles as prev_year_team_total_fumbles,
         -- opposing team previous year rushing/receiving stats
         opp_tsrm.rush_att as prev_year_opp_total_rush_att,
         opp_tsrm.rush_yds_per_att as prev_year_opp_total_rush_yds_per_att,
         opp_tsrm.rush_fd as prev_year_opp_total_rush_fd,
         opp_tsrm.rush_success as prev_year_opp_total_rush_success,
         opp_tsrm.rush_long as prev_year_opp_total_rush_long,
         opp_tsrm.rush_yds_per_g as prev_year_opp_total_rush_yds_per_g,
         opp_tsrm.rush_att_per_g as prev_year_opp_total_rush_att_per_g,
         opp_tsrm.rush_yds as prev_year_opp_total_rush_yds,
         opp_tsrm.rush_tds as prev_year_opp_total_rush_tds,
         opp_tsrm.targets as prev_year_opp_total_targets,
         opp_tsrm.rec as prev_year_opp_total_rec,
         opp_tsrm.rec_yds as prev_year_opp_total_rec_yds,
         opp_tsrm.rec_yds_per_rec as prev_year_opp_total_rec_yds_per_rec,
         opp_tsrm.rec_td as prev_year_opp_total_rec_td,
         opp_tsrm.rec_first_down as prev_year_opp_total_rec_first_down,
         opp_tsrm.rec_success as prev_year_opp_total_rec_success,
         opp_tsrm.rec_long as prev_year_opp_total_rec_long,
         opp_tsrm.rec_per_g as prev_year_opp_total_rec_per_g,
         opp_tsrm.rec_yds_per_g as prev_year_opp_total_rec_yds_per_g,
         opp_tsrm.catch_pct as prev_year_opp_total_catch_pct,
         opp_tsrm.rec_yds_per_tgt as prev_year_opp_total_rec_yds_per_tgt,
         opp_tsrm.touches as prev_year_opp_total_touches,
         opp_tsrm.yds_per_touch as prev_year_opp_total_yds_per_touch,
         opp_tsrm.yds_from_scrimmage as prev_year_opp_total_yds_from_scrimmage,
         opp_tsrm.rush_receive_td as prev_year_opp_total_rush_receive_td,
         opp_tsrm.fumbles as prev_year_opp_total_fumbles,
		 -- players team previous year kicking stats 
		 tskm.team_total_fg_long as prev_year_team_total_fg_long,
         tskm.team_total_fg_pct as prev_year_team_total_fg_pct,
         tskm.team_total_xpa as prev_year_team_total_xpa,
         tskm.team_total_xpm as prev_year_team_total_xpm,
         tskm.team_total_xp_pct as prev_year_team_total_xp_pct,
         tskm.team_total_kickoff_yds as prev_year_team_total_kickoff_yds,
         tskm.team_total_kickoff_tb_pct as prev_year_team_total_kickoff_tb_pct,
         -- opposing team previous year kicking stats
         opp_tskm.team_total_fg_long as prev_year_opp_total_fg_long,
         opp_tskm.team_total_fg_pct as prev_year_opp_total_fg_pct,
         opp_tskm.team_total_xpa as prev_year_opp_total_xpa,
         opp_tskm.team_total_xpm as prev_year_opp_total_xpm,
         opp_tskm.team_total_xp_pct as prev_year_opp_total_xp_pct,
         opp_tskm.team_total_kickoff_yds as prev_year_opp_total_kickoff_yds,
         opp_tskm.team_total_kickoff_tb_pct as prev_year_opp_total_kickoff_tb_pct,
		 -- players team previous year punting stats 
		 tspuntingmetrics.team_total_punt as prev_year_team_total_punt,
         tspuntingmetrics.team_total_punt_yds as prev_year_team_total_punt_yds,
         tspuntingmetrics.team_total_punt_yds_per_punt as prev_year_team_total_punt_yds_per_punt,
         tspuntingmetrics.team_total_punt_ret_yds_opp as prev_year_team_total_punt_ret_yds_opp,
         tspuntingmetrics.team_total_punt_net_yds as prev_year_team_total_punt_net_yds,
         tspuntingmetrics.team_total_punt_net_yds_per_punt as prev_year_team_total_punt_net_yds_per_punt,
         tspuntingmetrics.team_total_punt_long as prev_year_team_total_punt_long,
         tspuntingmetrics.team_total_punt_tb as prev_year_team_total_punt_tb,
         tspuntingmetrics.team_total_punt_tb_pct as prev_year_team_total_punt_tb_pct,
         tspuntingmetrics.team_total_punt_in_20 as prev_year_team_total_punt_in_20,
         tspuntingmetrics.team_total_punt_in_20_pct as prev_year_team_total_punt_in_20_pct,
         -- opposing team previous year punting stats
         opp_tspuntingmetrics.team_total_punt as prev_year_opp_total_punt,
         opp_tspuntingmetrics.team_total_punt_yds as prev_year_opp_total_punt_yds,
         opp_tspuntingmetrics.team_total_punt_yds_per_punt as prev_year_opp_total_punt_yds_per_punt,
         opp_tspuntingmetrics.team_total_punt_ret_yds_opp as prev_year_opp_total_punt_ret_yds_opp,
         opp_tspuntingmetrics.team_total_punt_net_yds as prev_year_opp_total_punt_net_yds,
         opp_tspuntingmetrics.team_total_punt_net_yds_per_punt as prev_year_opp_total_punt_net_yds_per_punt,
         opp_tspuntingmetrics.team_total_punt_long as prev_year_opp_total_punt_long,
         opp_tspuntingmetrics.team_total_punt_tb as prev_year_opp_total_punt_tb,
         opp_tspuntingmetrics.team_total_punt_tb_pct as prev_year_opp_total_punt_tb_pct,
         opp_tspuntingmetrics.team_total_punt_in_20 as prev_year_opp_total_punt_in_20,
         opp_tspuntingmetrics.team_total_punt_in_20_pct as prev_year_opp_total_punt_in_20_pct,
		 -- players team previous year scoring stats 
		 tssm.rush_td as prev_year_rush_td,
         tssm.rec_td as prev_year_rec_td,
         tssm.punt_ret_td as prev_year_punt_ret_td,
         tssm.kick_ret_td as prev_year_kick_ret_td,
         tssm.fumbles_rec_td as prev_year_fumbles_rec_td,
         tssm.def_int_td as prev_year_def_int_td,
         tssm.other_td as prev_year_other_td,
         tssm.total_td as prev_year_total_td,
         tssm.two_pt_md as prev_year_two_pt_md,
         tssm.def_two_pt as prev_year_def_two_pt,
         tssm.xpm as prev_year_xpm,
         tssm.xpa as prev_year_xpa,
         tssm.fgm as prev_year_fgm,
         tssm.fga as prev_year_fga,
         tssm.safety_md as prev_year_safety_md,
         tssm.scoring as prev_year_scoring,
         -- opposing team previous year scoring stats
         opp_tssm.rush_td as prev_year_opp_rush_td,
         opp_tssm.rec_td as prev_year_opp_rec_td,
         opp_tssm.punt_ret_td as prev_year_opp_punt_ret_td,
         opp_tssm.kick_ret_td as prev_year_opp_kick_ret_td,
         opp_tssm.fumbles_rec_td as prev_year_opp_fumbles_rec_td,
         opp_tssm.def_int_td as prev_year_opp_def_int_td,
         opp_tssm.other_td as prev_year_opp_other_td,
         opp_tssm.total_td as prev_year_opp_total_td,
         opp_tssm.two_pt_md as prev_year_opp_two_pt_md,
         opp_tssm.def_two_pt as prev_year_opp_def_two_pt,
         opp_tssm.xpm as prev_year_opp_xpm,
         opp_tssm.xpa as prev_year_opp_xpa,
         opp_tssm.fgm as prev_year_opp_fgm,
         opp_tssm.fga as prev_year_opp_fga,
         opp_tssm.safety_md as prev_year_opp_safety_md,
         opp_tssm.scoring as prev_year_opp_scoring,
	     -- players team previous year seasonal offensive rankings 
		 tsr.off_points as prev_year_off_points,
         tsr.off_total_yards as prev_year_off_total_yards,
         tsr.off_turnovers as prev_year_off_turnovers,
         tsr.off_fumbles_lost as prev_year_off_fumbles_lost,
         tsr.off_first_down as prev_year_off_first_down,
         tsr.off_pass_att as prev_year_off_pass_att,
         tsr.off_pass_yds as prev_year_off_pass_yds,
         tsr.off_pass_td as prev_year_off_pass_td,
         tsr.off_pass_int as prev_year_off_pass_int,
         tsr.off_pass_net_yds_per_att as prev_year_off_pass_net_yds_per_att,
         tsr.off_rush_att as prev_year_off_rush_att,
         tsr.off_rush_yds as prev_year_off_rush_yds,
         tsr.off_rush_td as prev_year_off_rush_td,
         tsr.off_rush_yds_per_att as prev_year_off_rush_yds_per_att,
         tsr.off_score_pct as prev_year_off_score_pct,
         tsr.off_turnover_pct as prev_year_off_turnover_pct,
         tsr.off_start_avg as prev_year_off_start_avg,
         tsr.off_time_avg as prev_year_off_time_avg,
         tsr.off_plays_per_drive as prev_year_off_plays_per_drive,
         tsr.off_yds_per_drive as prev_year_off_yds_per_drive,
         tsr.off_points_avg as prev_year_off_points_avg,
         tsr.off_third_down_pct as prev_year_off_third_down_pct,
         tsr.off_fourth_down_pct as prev_year_off_fourth_down_pct,
         tsr.off_red_zone_pct as prev_year_off_red_zone_pct,
         -- opposing team previous year seasonal offensive rankings 
		 opp_tsr.off_points as prev_year_opp_off_points,
         opp_tsr.off_total_yards as prev_year_opp_off_total_yards,
         opp_tsr.off_turnovers as prev_year_opp_off_turnovers,
         opp_tsr.off_fumbles_lost as prev_year_opp_off_fumbles_lost,
         opp_tsr.off_first_down as prev_year_opp_off_first_down,
         opp_tsr.off_pass_att as prev_year_opp_off_pass_att,
         opp_tsr.off_pass_yds as prev_year_opp_off_pass_yds,
         opp_tsr.off_pass_td as prev_year_opp_off_pass_td,
         opp_tsr.off_pass_int as prev_year_opp_off_pass_int,
         opp_tsr.off_pass_net_yds_per_att as prev_year_opp_off_pass_net_yds_per_att,
         opp_tsr.off_rush_att as prev_year_opp_off_rush_att,
         opp_tsr.off_rush_yds as prev_year_opp_off_rush_yds,
         opp_tsr.off_rush_td as prev_year_opp_off_rush_td,
         opp_tsr.off_rush_yds_per_att as prev_year_opp_off_rush_yds_per_att,
         opp_tsr.off_score_pct as prev_year_opp_off_score_pct,
         opp_tsr.off_turnover_pct as prev_year_opp_off_turnover_pct,
         opp_tsr.off_start_avg as prev_year_opp_off_start_avg,
         opp_tsr.off_time_avg as prev_year_opp_off_time_avg,
         opp_tsr.off_plays_per_drive as prev_year_opp_off_plays_per_drive,
         opp_tsr.off_yds_per_drive as prev_year_opp_off_yds_per_drive,
         opp_tsr.off_points_avg as prev_year_opp_off_points_avg,
         opp_tsr.off_third_down_pct as prev_year_opp_off_third_down_pct,
         opp_tsr.off_fourth_down_pct as prev_year_opp_off_fourth_down_pct,
         opp_tsr.off_red_zone_pct as prev_year_opp_off_red_zone_pct,
         -- players team previous year seasonal defensive rankings
         tsr.def_points as prev_year_team_def_points,
         tsr.def_total_yards as prev_year_team_def_total_yards,
         tsr.def_turnovers as prev_year_team_def_turnovers,
         tsr.def_fumbles_lost as prev_year_team_def_fumbles_lost,
         tsr.def_first_down as prev_year_team_def_first_down,
         tsr.def_pass_att as prev_year_team_def_pass_att,
         tsr.def_pass_yds as prev_year_team_def_pass_yds,
         tsr.def_pass_td as prev_year_team_def_pass_td,
         tsr.def_pass_int as prev_year_team_def_pass_int,
         tsr.def_pass_net_yds_per_att as prev_year_team_def_pass_net_yds_per_att,
         tsr.def_rush_att as prev_year_team_def_rush_att,
         tsr.def_rush_yds as prev_year_team_def_rush_yds,
         tsr.def_rush_td as prev_year_team_def_rush_td,
         tsr.def_rush_yds_per_att as prev_year_team_def_rush_yds_per_att,
         tsr.def_score_pct as prev_year_team_def_score_pct,
         tsr.def_turnover_pct as prev_year_team_def_turnover_pct,
         tsr.def_start_avg as prev_year_team_def_start_avg,
         tsr.def_time_avg as prev_year_team_def_time_avg,
         tsr.def_plays_per_drive as prev_year_team_def_plays_per_drive,
         tsr.def_yds_per_drive as prev_year_team_def_yds_per_drive,
         tsr.def_points_avg as prev_year_team_def_points_avg,
         tsr.def_third_down_pct as prev_year_team_def_third_down_pct,
         tsr.def_fourth_down_pct as prev_year_team_def_fourth_down_pct,
		 -- opposing teams previous year seasonal defensive rankings
         opp_tsr.def_points as prev_year_def_points,
         opp_tsr.def_total_yards as prev_year_def_total_yards,
         opp_tsr.def_turnovers as prev_year_def_turnovers,
         opp_tsr.def_fumbles_lost as prev_year_def_fumbles_lost,
         opp_tsr.def_first_down as prev_year_def_first_down,
         opp_tsr.def_pass_att as prev_year_def_pass_att,
         opp_tsr.def_pass_yds as prev_year_def_pass_yds,
         opp_tsr.def_pass_td as prev_year_def_pass_td,
         opp_tsr.def_pass_int as prev_year_def_pass_int,
         opp_tsr.def_pass_net_yds_per_att as prev_year_def_pass_net_yds_per_att,
         opp_tsr.def_rush_att as prev_year_def_rush_att,
         opp_tsr.def_rush_yds as prev_year_def_rush_yds,
         opp_tsr.def_rush_td as prev_year_def_rush_td,
         opp_tsr.def_rush_yds_per_att as prev_year_def_rush_yds_per_att,
         opp_tsr.def_score_pct as prev_year_def_score_pct,
         opp_tsr.def_turnover_pct as prev_year_def_turnover_pct,
         opp_tsr.def_start_avg as prev_year_def_start_avg,
         opp_tsr.def_time_avg as prev_year_def_time_avg,
         opp_tsr.def_plays_per_drive as prev_year_def_plays_per_drive,
         opp_tsr.def_yds_per_drive as prev_year_def_yds_per_drive,
         opp_tsr.def_points_avg as prev_year_def_points_avg,
         opp_tsr.def_third_down_pct as prev_year_def_third_down_pct,
         opp_tsr.def_fourth_down_pct as prev_year_def_fourth_down_pct,
         opp_tsr.def_red_zone_pct as prev_year_def_red_zone_pct,
		 -- opposing teams previous year defensive metrics 
		 opp_tsdm.points as prev_year_opp_def_points,
         opp_tsdm.total_yards as prev_year_opp_def_total_yards,
         opp_tsdm.plays_offense as prev_year_opp_def_plays_offense,
         opp_tsdm.yds_per_play_offense as prev_year_opp_def_yds_per_play_offense,
         opp_tsdm.turnovers as prev_year_opp_def_turnovers,
         opp_tsdm.fumbles_lost as prev_year_opp_def_fumbles_lost,
         opp_tsdm.first_down as prev_year_opp_def_first_down,
         opp_tsdm.pass_cmp as prev_year_opp_def_pass_cmp,
         opp_tsdm.pass_att as prev_year_opp_def_pass_att,
         opp_tsdm.pass_yds as prev_year_opp_def_pass_yds,
         opp_tsdm.pass_td as prev_year_opp_def_pass_td,
         opp_tsdm.pass_int as prev_year_opp_def_pass_int,
         opp_tsdm.pass_net_yds_per_att as prev_year_opp_def_pass_net_yds_per_att,
         opp_tsdm.pass_fd as prev_year_opp_def_pass_fd,
         opp_tsdm.rush_att as prev_year_opp_def_rush_att,
         opp_tsdm.rush_yds as prev_year_opp_def_rush_yds,
         opp_tsdm.rush_td as prev_year_opp_def_rush_td,
         opp_tsdm.rush_yds_per_att as prev_year_opp_def_rush_yds_per_att,
         opp_tsdm.rush_fd as prev_year_opp_def_rush_fd,
         opp_tsdm.penalties as prev_year_opp_def_penalties,
         opp_tsdm.penalties_yds as prev_year_opp_def_penalties_yds,
         opp_tsdm.pen_fd as prev_year_opp_def_pen_fd,
         opp_tsdm.drives as prev_year_opp_def_drives,
         opp_tsdm.score_pct as prev_year_opp_def_score_pct,
         opp_tsdm.turnover_pct as prev_year_opp_def_turnover_pct,
         opp_tsdm.start_avg as prev_year_opp_def_start_avg,
         opp_tsdm.time_avg as prev_year_opp_def_time_avg,
         opp_tsdm.plays_per_drive as prev_year_opp_def_plays_per_drive,
         opp_tsdm.yds_per_drive as prev_year_opp_def_yds_per_drive,
         opp_tsdm.points_avg as prev_year_opp_def_points_avg,
         opp_tsdm.third_down_att as prev_year_opp_def_third_down_att,
         opp_tsdm.third_down_success as prev_year_opp_def_third_down_success,
         opp_tsdm.third_down_pct as prev_year_opp_def_third_down_pct,
         opp_tsdm.fourth_down_att as prev_year_opp_def_fourth_down_att,
         opp_tsdm.fourth_down_success as prev_year_opp_def_fourth_down_success,
         opp_tsdm.fourth_down_pct as prev_year_opp_def_fourth_down_pct,
         opp_tsdm.red_zone_att as prev_year_opp_def_red_zone_att,
         opp_tsdm.red_zone_scores as prev_year_opp_def_red_zone_scores,
         opp_tsdm.red_zone_pct as prev_year_opp_def_red_zone_pct,
         opp_tsdm.def_int as prev_year_opp_def_int,
         opp_tsdm.def_int_yds as prev_year_opp_def_int_yds,
         opp_tsdm.def_int_td as prev_year_opp_def_int_td,
         opp_tsdm.def_int_long as prev_year_opp_def_int_long,
         opp_tsdm.pass_defended as prev_year_opp_def_pass_defended,
         opp_tsdm.fumbles_forced as prev_year_opp_def_fumbles_forced,
         opp_tsdm.fumbles_rec as prev_year_opp_def_fumbles_rec,
         opp_tsdm.fumbles_rec_yds as prev_year_opp_def_fumbles_rec_yds,
         opp_tsdm.fumbles_rec_td as prev_year_opp_def_fumbles_rec_td,
         opp_tsdm.sacks as prev_year_opp_def_sacks,
         opp_tsdm.tackles_combined as prev_year_opp_def_tackles_combined,
         opp_tsdm.tackles_solo as prev_year_opp_def_tackles_solo,
         opp_tsdm.tackles_assists as prev_year_opp_def_tackles_assists,
         opp_tsdm.tackles_loss as prev_year_opp_def_tackles_loss,
         opp_tsdm.qb_hits as prev_year_opp_def_qb_hits,
         opp_tsdm.safety_md as prev_year_opp_def_safety_md,
         -- players team previous year defensive metrics
         tsdm.points as prev_year_team_def_points_metrics,
         tsdm.total_yards as prev_year_team_def_total_yards_metrics,
         tsdm.plays_offense as prev_year_team_def_plays_offense,
         tsdm.yds_per_play_offense as prev_year_team_def_yds_per_play_offense,
         tsdm.turnovers as prev_year_team_def_turnovers_metrics,
         tsdm.fumbles_lost as prev_year_team_def_fumbles_lost_metrics,
         tsdm.first_down as prev_year_team_def_first_down_metrics,
         tsdm.pass_cmp as prev_year_team_def_pass_cmp,
         tsdm.pass_att as prev_year_team_def_pass_att_metrics,
         tsdm.pass_yds as prev_year_team_def_pass_yds_metrics,
         tsdm.pass_td as prev_year_team_def_pass_td_metrics,
         tsdm.pass_int as prev_year_team_def_pass_int_metrics,
         tsdm.pass_net_yds_per_att as prev_year_team_def_pass_net_yds_per_att_metrics,
         tsdm.pass_fd as prev_year_team_def_pass_fd,
         tsdm.rush_att as prev_year_team_def_rush_att_metrics,
         tsdm.rush_yds as prev_year_team_def_rush_yds_metrics,
         tsdm.rush_td as prev_year_team_def_rush_td_metrics,
         tsdm.rush_yds_per_att as prev_year_team_def_rush_yds_per_att_metrics,
         tsdm.rush_fd as prev_year_team_def_rush_fd,
         tsdm.penalties as prev_year_team_def_penalties,
         tsdm.penalties_yds as prev_year_team_def_penalties_yds,
         tsdm.pen_fd as prev_year_team_def_pen_fd,
         tsdm.drives as prev_year_team_def_drives,
         tsdm.score_pct as prev_year_team_def_score_pct_metrics,
         tsdm.turnover_pct as prev_year_team_def_turnover_pct_metrics,
         tsdm.start_avg as prev_year_team_def_start_avg_metrics,
         tsdm.time_avg as prev_year_team_def_time_avg_metrics,
         tsdm.plays_per_drive as prev_year_team_def_plays_per_drive_metrics,
         tsdm.yds_per_drive as prev_year_team_def_yds_per_drive_metrics,
         tsdm.points_avg as prev_year_team_def_points_avg_metrics,
         tsdm.third_down_att as prev_year_team_def_third_down_att,
         tsdm.third_down_success as prev_year_team_def_third_down_success,
         tsdm.third_down_pct as prev_year_team_def_third_down_pct_metrics,
         tsdm.fourth_down_att as prev_year_team_def_fourth_down_att,
         tsdm.fourth_down_success as prev_year_team_def_fourth_down_success,
         tsdm.fourth_down_pct as prev_year_team_def_fourth_down_pct_metrics,
         tsdm.red_zone_att as prev_year_team_def_red_zone_att,
         tsdm.red_zone_scores as prev_year_team_def_red_zone_scores,
         tsdm.red_zone_pct as prev_year_team_def_red_zone_pct_metrics,
         tsdm.def_int as prev_year_team_def_int,
         tsdm.def_int_yds as prev_year_team_def_int_yds,
         tsdm.def_int_td as prev_year_team_def_int_td,
         tsdm.def_int_long as prev_year_team_def_int_long,
         tsdm.pass_defended as prev_year_team_def_pass_defended,
         tsdm.fumbles_forced as prev_year_team_def_fumbles_forced,
         tsdm.fumbles_rec as prev_year_team_def_fumbles_rec,
         tsdm.fumbles_rec_yds as prev_year_team_def_fumbles_rec_yds,
         tsdm.fumbles_rec_td as prev_year_team_def_fumbles_rec_td,
         tsdm.sacks as prev_year_team_def_sacks,
         tsdm.tackles_combined as prev_year_team_def_tackles_combined,
         tsdm.tackles_solo as prev_year_team_def_tackles_solo,
         tsdm.tackles_assists as prev_year_team_def_tackles_assists,
         tsdm.tackles_loss as prev_year_team_def_tackles_loss,
         tsdm.qb_hits as prev_year_team_def_qb_hits,
         tsdm.safety_md as prev_year_team_def_safety_md,
		 -- player passing stats from previous year
		 pssm.games_started,
		 pssm.pass_att,
		 pssm.pass_cmp_pct,
		 pssm.pass_yds,
		 pssm.pass_td,
		 pssm.pass_td_pct,
		 pssm.pass_int,
		 pssm.pass_int_pct,
		 pssm.pass_first_down,
		 pssm.pass_success,
		 pssm.pass_long,
		 pssm.pass_yds_per_att,
		 pssm.pass_adj_yds_per_att,
		 pssm.pass_yds_per_cmp,
		 pssm.pass_yds_per_g,
		 pssm.pass_rating,
		 pssm.qbr,
		 pssm.pass_sacked,
		 pssm.pass_sacked_yds,
		 pssm.pass_sacked_pct,
		 pssm.pass_net_yds_per_att,
		 pssm.pass_adj_net_yds_per_att,
		 pssm.comebacks,
		 pssm.game_winning_drives,
		 -- player rushing & receiving stats from previous year 
		 psrrm.games_started,
		 psrrm.rush_att,
		 psrrm.rush_yds_per_att,
		 psrrm.rush_fd,
		 psrrm.rush_success,
		 psrrm.rush_long,
		 psrrm.rush_yds_per_g,
		 psrrm.rush_att_per_g,
		 psrrm.rush_yds,
		 psrrm.rush_tds,
		 psrrm.targets,
		 psrrm.rec,
		 psrrm.rec_yds,
		 psrrm.rec_yds_per_rec,
		 psrrm.rec_td,
		 psrrm.rec_first_down,
		 psrrm.rec_success,
		 psrrm.rec_long,
		 psrrm.rec_per_g,
		 psrrm.rec_yds_per_g,
		 psrrm.catch_pct,
		 psrrm.rec_yds_per_tgt,
		 psrrm.touches,
		 psrrm.yds_per_touch,
		 psrrm.yds_from_scrimmage,
		 psrrm.rush_receive_td,
		 psrrm.fumbles, 
		 -- player scoring metrics from previous year 
		 player_seasonal_sm.rush_td,
		 player_seasonal_sm.rec_td,
		 player_seasonal_sm.punt_ret_td,
		 player_seasonal_sm.kick_ret_td,
		 player_seasonal_sm.fumbles_rec_td,
		 player_seasonal_sm.other_td,
		 player_seasonal_sm.total_td,
		 player_seasonal_sm.two_pt_md,
		 player_seasonal_sm.scoring,
         CASE
           WHEN tbo.favorite_team_id = t.team_id THEN 1
		 ELSE 0
           END AS is_favorited,
		 pp.props
      FROM
         player_game_log pgl -- player game logs (week to week games)
      JOIN 
         player_weekly_agg_metrics pam ON pgl.week = pam.week AND pgl.year = pam.season AND pgl.player_id = pam.player_id -- weekly aggregate metrics for player 
      JOIN
         player_depth_chart pdc ON pdc.week = pgl.week AND pgl.year = pdc.season AND pgl.player_id = pdc.player_id -- player depth chart position 
      JOIN 
         player p ON p.player_id = pgl.player_id -- player information 
      JOIN 
         player_teams pt ON p.player_id = pt.player_id AND pgl.week >= pt.strt_wk AND pgl.week <= pt.end_wk AND pt.season = pgl.year -- players 
      JOIN 
         player_demographics pd ON p.player_id = pd.player_id AND pgl.year = pd.season -- demographic metrics for player 
      JOIN 
         team t ON pt.team_id = t.team_id -- team the player is on
      JOIN 
         team td ON pgl.opp = td.team_id -- team the player is playing against 
      LEFT JOIN 
         player_seasonal_passing_metrics pssm ON p.player_id = pssm.player_id AND pssm.season = pgl.year AND pssm.team_id = t.team_id -- player seasonal passing metrics for previous year 
      LEFT JOIN 
         player_seasonal_rushing_receiving_metrics psrrm ON p.player_id = psrrm.player_id AND psrrm.season = pgl.year AND psrrm.team_id = t.team_id -- player seasonal rushing / receiving metrics for previous year 
      LEFT JOIN 
         player_seasonal_scoring_metrics player_seasonal_sm ON p.player_id = player_seasonal_sm.player_id AND player_seasonal_sm.season = pgl.year AND player_seasonal_sm.team_id = t.team_id -- player seasonal scoring metrics for previous year 
      LEFT JOIN
         team_seasonal_general_metrics tsgm ON t.team_id = tsgm.team_id AND (pgl.year - 1) = tsgm.season -- team general metrics for previous year
      LEFT JOIN
         team_seasonal_general_metrics opp_tsgm ON td.team_id = opp_tsgm.team_id AND (pgl.year - 1) = opp_tsgm.season -- opposing team general metrics for previous year
      LEFT JOIN
         team_seasonal_rushing_receiving_metrics tsrm ON t.team_id = tsrm.team_id AND (pgl.year - 1) = tsrm.season -- team rushing/receiving metrics for previous year
      LEFT JOIN
         team_seasonal_rushing_receiving_metrics opp_tsrm ON td.team_id = opp_tsrm.team_id AND (pgl.year - 1) = opp_tsrm.season -- opposing team rushing/receiving metrics for previous year
      LEFT JOIN 
         team_seasonal_passing_metrics tspassingmetrics ON t.team_id = tspassingmetrics.team_id AND (pgl.year - 1) = tspassingmetrics.season -- team passing metrics for previous year 
      LEFT JOIN 
         team_seasonal_passing_metrics opp_tspassingmetrics ON td.team_id = opp_tspassingmetrics.team_id AND (pgl.year - 1) = opp_tspassingmetrics.season -- opposing team passing metrics for previous year
      LEFT JOIN 
         team_seasonal_kicking_metrics tskm ON t.team_id = tskm.team_id AND (pgl.year - 1) = tskm.season -- team kicking metrics for previous year 
      LEFT JOIN 
         team_seasonal_kicking_metrics opp_tskm ON td.team_id = opp_tskm.team_id AND (pgl.year - 1) = opp_tskm.season -- opposing team kicking metrics for previous year
      LEFT JOIN 
         team_seasonal_punting_metrics tspuntingmetrics ON t.team_id = tspuntingmetrics.team_id AND (pgl.year - 1) = tspuntingmetrics.season -- team punting metrics for previous year 
      LEFT JOIN 
         team_seasonal_punting_metrics opp_tspuntingmetrics ON td.team_id = opp_tspuntingmetrics.team_id AND (pgl.year - 1) = opp_tspuntingmetrics.season -- opposing team punting metrics for previous year
      LEFT JOIN 
         team_seasonal_scoring_metrics tssm ON t.team_id = tssm.team_id AND (pgl.year - 1) = tssm.season -- team scoring metrics for previous years 
      LEFT JOIN 
         team_seasonal_scoring_metrics opp_tssm ON td.team_id = opp_tssm.team_id AND (pgl.year - 1) = opp_tssm.season -- opposing team scoring metrics for previous years
      LEFT JOIN 
         team_seasonal_defensive_metrics tsdm ON t.team_id = tsdm.team_id AND (pgl.year - 1) = tsdm.season -- team defensive metrics for previous year
      LEFT JOIN
         team_seasonal_defensive_metrics opp_tsdm ON td.team_id = opp_tsdm.team_id AND (pgl.year - 1) = opp_tsdm.season -- opposing team defensive metrics for previous year
      LEFT JOIN 
         team_seasonal_ranks tsr ON t.team_id = tsr.team_id AND (pgl.year - 1) = tsr.season -- team seasonal ranks for previous year 
      LEFT JOIN
         team_seasonal_ranks opp_tsr ON td.team_id = opp_tsr.team_id AND (pgl.year - 1) = opp_tsr.season -- opposing team seasonal ranks for previous year
      JOIN 
         team_ranks t_tr ON t.team_id = t_tr.team_id AND pgl.week = t_tr.week AND pgl.year = t_tr.season -- players team weekly rankings heading into matchup
      JOIN
         team_ranks t_td ON td.team_id = t_td.team_id AND pgl.week = t_td.week AND pgl.year = t_td.season -- opposing team weekly rankings heading into matchup
      JOIN
         PlayerProps pp ON p.name = pp.player_name AND pgl.week = pp.week AND pgl.year = pp.season -- player betting lines 
      JOIN 
         team_betting_odds tbo -- team betting lines 
      ON (
        (pgl.home_team = TRUE AND tbo.home_team_id = t.team_id AND tbo.away_team_id = td.team_id AND pgl.week = tbo.week) 
            OR 
        (pgl.home_team = FALSE AND tbo.away_team_id = t.team_id AND tbo.home_team_id = td.team_id AND pgl.week = tbo.week)
      ) 

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
TODO: FIX ME SO THAT OFFENSIVE/DEFENSIVE RANKS AREN'T THE SAME 

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
            pbo.player_name = '%s'
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
        t_tr.off_rush_rank,
        t_tr.off_pass_rank,
        t_td.def_rush_rank,
        t_td.def_pass_rank,
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
	  	team_ranks t_tr ON t.team_id = t_tr.team_id AND pgl.week = t_tr.week AND pgl.year = t_tr.season
    JOIN 
        team df ON pgl.opp = df.team_id
	JOIN
	  	team_ranks t_td ON df.team_id = t_td.team_id AND pgl.week = t_td.week AND pgl.year = t_td.season
    JOIN
        team_betting_odds tbo 
    ON ((tbo.home_team_id = t.team_id OR tbo.away_team_id = t.team_id) 
       AND (tbo.home_team_id = df.team_id OR tbo.away_team_id = df.team_id))
    JOIN (
        SELECT 
            pgl.player_id, 
            AVG(pgl.fantasy_points) AS avg_fantasy_points
            FROM player_game_log pgl
            WHERE pgl.player_id = (SELECT player_id FROM player WHERE name = '%s')
            AND pgl.year = %s
            GROUP BY pgl.player_id
    ) player_avg 
    ON 
        player_avg.player_id = pgl.player_id
    JOIN PlayerProps pp 
        ON pp.player_name = p.name AND pp.week = tbo.week AND pp.season = tbo.season
    WHERE 
        p.name = '%s' AND
        tbo.week = %s AND tbo.season = %s
    GROUP BY 
        p.position, t.off_rush_rank, t.off_pass_rank, df.def_rush_rank, df.def_pass_rank, 
        tbo.game_over_under, tbo.spread, tbo.favorite_team_id, player_avg.avg_fantasy_points, is_favorited, pp.props,
		t_tr.off_rush_rank, t_tr.off_pass_rank, t_td.def_rush_rank, t_td.def_pass_rank;
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
    

def fetch_players_active_in_one_of_specified_seasons(start_year: int, end_year: int):
    """
    Retrieve players who were on an active roster in at least one of the specified seasons.

    Args:
        start_year (int): The starting season year.
        end_year (int): The ending season year.

    Returns:
        list: List of distinct players.
    """
    
    sql = """
        SELECT DISTINCT
            p.player_id,
            p.name,
            p.position
        FROM 
            player p 
        JOIN 
            player_teams pt 
        ON
            p.player_id = pt.player_id 
        WHERE 
            pt.season BETWEEN %s AND %s
    """
    
    players = []

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (start_year, end_year))
            rows = cur.fetchall()
            
            for row in rows:
                players.append({"player_id": row[0], "player_name": row[1], "position": row[2]})
    
    except Exception as e:
        logging.error(
            f"An error occurred while fetching players active between seasons {start_year} and {end_year}: {e}"
        )
        raise e

    return players


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
            p.position
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
                players.append({"player_id": row[0], "player_name": row[1], "position": row[2]})
                


    except Exception as e:
        logging.error(
            f"An error occurred while fetching players on active roster in season {year}: {e}"
        )
        raise e

    return players


"""
Retrieve the latest week we have persisted data in our 'team_betting_odds' table 

Args:
   year (int): season to retrieve data for 

Return:
   week (int): the latest week persisted in our db table
"""


def fetch_max_week_persisted_in_team_betting_odds_table(year: int):
    sql = "SELECT week FROM team_betting_odds WHERE season = %s AND week = (SELECT MAX(week) FROM team_betting_odds WHERE season = %s)"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (year,year))
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
            cur.execute(sql, (record["player_id"], record["team_id"], record['season'], record["strt_wk"], record["end_wk"]))
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


def fetch_player_fantasy_points(player_id: int, season: int, end_week: int): 
    """
    Functionality to retrieve players fantasy points from beginning of season to specific week

    Args:
        player_id (int): player ID to fetch fantasy points for 
        season (int): season corresponding to fantasy points 
        end_week (int): week to retrieve points up to 
    
    Return: 
        fantasy_ponts (list): list of fantasy points 
    """

    sql = " \
        SELECT week, fantasy_points \
        FROM player_game_log WHERE year = %s AND week >= 1 AND week <= %s AND player_id = %s \
        ORDER BY week \
    "

    try:
        connection = get_connection()
        
        fantasy_points = [] 
        with connection.cursor() as cur:
            cur.execute(sql, (season, end_week, player_id))
            rows = cur.fetchall()
            
            for row in rows: 
                fantasy_points.append({"week": row[0], "fantasy_points": row[1]})
            
            return fantasy_points

    except Exception as e:
        logging.error(
            f"An error occurred while fetching fantasy points corresponding season {season} and end week {end_week}: {e}"
        )
        raise e


def fetch_player_depth_chart_record_by_pk(record: dict): 
    """
    Retrieve player_depth_chart record by PK 

    Args:
        record (dict): mapping of PK's 
    
    Returns:
        record (dict): record in DB 
    """

    sql = "SELECT * FROM player_depth_chart WHERE player_id = %s AND season = %s AND week = %s"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (record["player_id"], record['season'], record["week"]))
            row = cur.fetchone()

            if row:
                return row[0]
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving player depth chart: {e}"
        )
        raise e


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
            f"An error occurred while retrieving player_demographic record pertaining to season {season} and player_id {player_id}", exc_info=True
        )
        raise e


def fetch_teams_home_away_wins_and_losses(season: int, team_id: int):
    """
    Functionality to retrieve teams home and away wins and losses for a given season and team
    
    Args:
        season (int): season to retrieve data for
        team_id (int): team ID to retrieve data for
    
    Returns:
        dict: Dictionary containing:
            wins (int): number of wins
            losses (int): number of losses  
            home_wins (int): number of home wins
            home_losses (int): number of home losses
            away_wins (int): number of away wins 
            away_losses (int): number of away losses
            win_pct (float): win percentage
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
                    'wins': row[0],
                    'losses': row[1], 
                    'home_wins': row[2],
                    'home_losses': row[3],
                    'away_wins': row[4],
                    'away_losses': row[5],
                    'win_pct': row[6]
                }
            else:
                return None

    except Exception as e:
        logging.error(
            f"An error occurred while fetching teams home and away wins and losses for season {season} and team {team_id}: {e}"
        )
        raise e

    
    
    

