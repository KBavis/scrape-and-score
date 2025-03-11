from .connection import get_connection, close_connection
import logging
import pandas as pd

"""
Functionality to persist a particular player 

Args: 
   players (list): list of players to persist 
"""


def insert_players(players: list):
    try:
        for player in players:
            insert_player(player)

    except Exception as e:
        logging.error(
            f"An exception occurred while attempting to insert players {players}",
            exc_info=True,
        )
        raise e


"""
Functionality to persist a single player 

Args: 
   player (dict): player to insert into our db 
"""


def insert_player(player: dict):
    query = """
      INSERT INTO player (name, position) 
      VALUES (%s, %s)
   """

    try:
        player_name = player["name"]
        player_position = player["position"]

        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(
                query, (player_name, player_position)
            )  
            
            # Commit the transaction to persist data
            connection.commit()
            logging.info(
                f"Successfully inserted player {player_name} into the database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the player {player}", exc_info=True
        )
        raise e
    


"""
Functionality to persist mutliple teams into our db 

Args: 
   teams (list): list of teams to insert into our db 
Returns: 
   teams (list): mapping of team names and ids we inserted

"""


def insert_teams(teams: list):
    team_mappings = []

    try:
        for team in teams:
            team_id = insert_team(team)
            team_mappings.append({"team_id": team_id, "name": team})

        return team_mappings

    except Exception as e:
        logging.error(
            f"An exception occured while inserting the following teams into our db: {teams}: {e}"
        )
        raise e


"""
Functionality to persist a single team into our db 

Args: 
   team_name (str): team to insert into our db 

Returns: 
   team_id (int): id corresponding to a particular team 
"""


def insert_team(team_name: str):
    sql = "INSERT INTO team (name) VALUES (%s) RETURNING team_id"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (team_name,))  # ensure team name is a tuple

            rows = cur.fetchone()
            if rows:
                team_id = rows[0]

            connection.commit()
            return team_id

    except Exception as e:
        logging.error(
            f"An exception occured while inserting the following team '{team_name}' into our db: {e}"
        )
        raise e


"""
Functionality to persist multiple game logs for a team 

Args:
   game_logs (list): list of tuples to insert into team_game_logs 
   
Returns:
   None
"""


def insert_team_game_logs(game_logs: list):
    sql = """
      INSERT INTO team_game_log (team_id, week, day, year, rest_days, home_team, distance_traveled, opp, result, points_for, points_allowed, tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
   """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, game_logs)

            connection.commit()
            logging.info(
                f"Successfully inserted {len(game_logs)} team game logs into the database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the team game logs: {game_logs}",
            exc_info=True,
        )
        raise e


"""
Functionality to persist multiple game logs for a RB 

Args:
   game_logs (list): list of tuples to insert into player_game_logs 

Returns: 
   None
"""


def insert_rb_player_game_logs(game_logs: list):
    sql = """
      INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, rush_att, rush_yds, rush_tds, tgt, rec, rec_yd, rec_td)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
   """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, game_logs)

            connection.commit()
            logging.info(
                f"Successfully inserted {len(game_logs)} RB game logs into the database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the RB game logs: {game_logs}",
            exc_info=True,
        )
        raise e


"""
Functionality to persist multiple game logs for a QB

Args:
   game_logs (list): list of tuples to insert into player_game_logs 

Returns: 
   None
"""


def insert_qb_player_game_logs(game_logs: list):
    sql = """
         INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, completions, attempts, pass_yd, pass_td, interceptions, rating, sacked, rush_att, rush_yds, rush_tds)
         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
   """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, game_logs)

            connection.commit()
            logging.info(
                f"Successfully inserted {len(game_logs)} QB game logs into the database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the QB game logs: {game_logs}",
            exc_info=True,
        )
        raise e


"""
Functionality to persist multiple game logs for a WR or TE

Args:
   game_logs (list): list of tuples to insert into player_game_logs 

Returns: 
   None
"""


def insert_wr_or_te_player_game_logs(game_logs: list):
    sql = """
      INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, tgt, rec, rec_yd, rec_td, snap_pct)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
   """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, game_logs)

            connection.commit()
            logging.info(
                f"Successfully inserted {len(game_logs)} WR/TE game logs into the database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the WR/TE game logs: {game_logs}",
            exc_info=True,
        )
        raise e


"""
Functionality to update player game log with calculated fantasy points

Args:
   fantasy_points (list): list of items containing player_game_log PK and their fantasy points

Returns: 
   None
"""


def add_fantasy_points(fantasy_points: list):
    sql = "UPDATE player_game_log SET fantasy_points = %s WHERE player_id = %s AND week = %s AND year = %s"

    try:
        connection = get_connection()

        params = [
            (log["fantasy_points"], log["player_id"], log["week"], log["year"])
            for log in fantasy_points
        ]  # create needed tuples for executemany functionality

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(fantasy_points)} players fantasy points into the database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the follwoing fantasy points: {fantasy_points}",
            exc_info=True,
        )
        raise e


"""
Insert record into 'team_ranks' with newly determined rankings 

Args:
   rankings (list): list of rankings to persist 
"""


def insert_team_rankings(rankings: list):
    sql = f"""
            INSERT INTO team_ranks (off_rush_rank, off_pass_rank, def_pass_rank, def_rush_rank, week, season, team_id) 
            VALUES (%s, %s, %s, %s, %s, %s, %s) 
        """

    try:
        connection = get_connection()

        params = [
            (rank["off_rush_rank"], rank["off_pass_rank"], rank["def_pass_rank"], rank["def_rush_rank"], rank['week'], rank['season'], rank["team_id"]) for rank in rankings
        ]  # create tuple needed to update records

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(rankings)} rankings in the database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following rankings: {rankings}",
            exc_info=True,
        )
        raise e


"""
Functionality to insert historical player props into our DB 

Args:
    player_props (dict):  relevant season long player props to persist
    season (int): relevant season

Returns:
    None
"""


def insert_player_props(player_props: dict, season: int):
    sql = """
        INSERT INTO player_betting_odds (player_id, player_name, label, cost, line, week, season) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    player_id = player_props['player_id']
    player_name = player_props['player_name']
    
    params = [
        (
            player_id,
            player_name,
            odds['label'],
            odds['cost'],
            odds['line'],
            week_data['week'],
            season
        )
        for week_data in player_props['season_odds']
        for odds in week_data['week_odds']
    ]
    
    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(params)} player historical odds into our database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following historical player odds: {player_props}",
            exc_info=True,
        )
        raise e
    
    


"""
Functionality to insert historical betting odds into our DB 

Args:
   betting_odds (list): list of historical betting odds 

Returns:
   None
"""


def insert_teams_odds(betting_odds: list, upcoming=False):
    if not upcoming:
        sql = """
         INSERT INTO team_betting_odds (home_team_id, away_team_id, home_team_score, away_team_score, week, season, game_over_under, favorite_team_id, spread, total_points, over_hit, under_hit, favorite_covered, underdog_covered)
         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
      """
        params = [
            (
                odds["home_team_id"],
                odds["away_team_id"],
                odds["home_team_score"],
                odds["away_team_score"],
                odds["week"],
                odds["year"],
                odds["game_over_under"],
                odds["favorite_team_id"],
                odds["spread"],
                odds["total_points"],
                odds["over_hit"],
                odds["under_hit"],
                odds["favorite_covered"],
                odds["underdog_covered"],
            )
            for odds in betting_odds
        ]
    else:
        sql = """
         INSERT INTO team_betting_odds (home_team_id, away_team_id, week, season, game_over_under, favorite_team_id, spread)
         VALUES (%s, %s, %s, %s, %s, %s, %s)
      """
        params = [
            (
                odds["home_team_id"],
                odds["away_team_id"],
                odds["week"],
                odds["year"],
                odds["game_over_under"],
                odds["favorite_team_id"],
                odds["spread"],
            )
            for odds in betting_odds
        ]

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(betting_odds)} team {'historical' if not upcoming else 'upcoming'} odds into our database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following  {'historical' if not upcoming else 'upcoming'} team odds: {betting_odds}",
            exc_info=True,
        )
        raise e


"""
   Update exisiting team_betting_odds records with the outcomes of each game
   
   Args:
      update_records (list): list of records to update 
   
   Returns:
      None
"""


def update_team_betting_odds_records_with_outcomes(update_records: list):
    sql = f"""
            UPDATE team_betting_odds
            SET 
               home_team_score = %s,
               away_team_score = %s, 
	            total_points = %s, 
	            over_hit = %s,
	            under_hit = %s,
	            favorite_covered = %s, 
	            underdog_covered = %s
            WHERE 
               home_team_id = %s 
               AND 
               away_team_id = %s
               AND 
               season = %s
               AND 
               week = %s
         """

    try:
        connection = get_connection()

        params = [
            (
                record["home_team_score"],
                record["visit_team_score"],
                record["total_points"],
                record["over_hit"],
                record["under_hit"],
                record["favorite_covered"],
                record["underdog_covered"],
                record["home_team_id"],
                record["visit_team_id"],
                record["year"],
                record["week"],
            )
            for record in update_records
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully updated {len(update_records)} team betting odds records in the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while updating the following team_betting_odds: {update_records}",
            exc_info=True,
        )
        raise e



def insert_bye_week_rankings(team_bye_weeks: list, season: int, ): 
    sql = f"""
            INSERT INTO team_ranks (team_id, week, season, off_rush_rank, off_pass_rank, def_rush_rank, def_pass_rank)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
         """

    try:
        connection = get_connection()

        params = [
            (
                record["team_id"],
                record["week"],
                season,
                -1, 
                -1,
                -1,
                -1
            )
            for record in team_bye_weeks 
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(team_bye_weeks)} team bye week rankings into the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following team bye week rankings into team_ranks: {team_bye_weeks}",
            exc_info=True,
        )
        raise e



def insert_player_teams_records(player_teams_records: list): 
    sql = f"""
            INSERT INTO player_teams (player_id, team_id, season, strt_wk, end_wk)
            VALUES (%s, %s, %s, %s, %s)
         """

    try:
        connection = get_connection()

        params = [
            (
                record["player_id"],
                record["team_id"],
                record["season"],
                record["strt_wk"],
                record["end_wk"]
            )
            for record in player_teams_records 
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(player_teams_records)} player_teams records into the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_teams records into our Databse: {player_teams_records}",
            exc_info=True,
        )
        raise e


def insert_player_aggregate_metrics(player_agg_metrics: list): 
    sql = f"""
            INSERT INTO player_aggregate_metrics (player_id, week, season, fantasy_points)
            VALUES (%s, %s, %s, %s)
         """

    try:
        connection = get_connection()

        params = [
            (
                record["player_id"],
                record["week"],
                record["season"],
                record["fantasy_points"]
            )
            for record in player_agg_metrics 
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(player_agg_metrics)} player_aggregate_metrics records into the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_aggregate_metrics records into our db: {player_agg_metrics}",
            exc_info=True,
        )
        raise e