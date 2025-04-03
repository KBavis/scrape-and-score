from .connection import get_connection, close_connection
import logging
import pandas as pd
from service import team_service, player_service
import time
from . import fetch_data

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


def update_player_hashed_name(hashed_names: list): 
    """
    Update player records with a players hashed name (href for pro-football-refernece player pages)

    Args:
        hashed_names (list): list of dictionary elements containing 'player_id' and 'hashed_name' 
    """
    query = """
        UPDATE player 
        SET hashed_name = %s 
        WHERE player_id = %s
    """


    try:
        params = [(player["hashed_name"], player["player_id"]) for player in hashed_names]

        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(
                query, params
            )  
            
            # Commit the transaction to persist data
            connection.commit()
            logging.info(
                f"Successfully updated {len(params)} player records with hashed names"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while updating player records with hashed names", exc_info=True
        )
        raise e



def update_player_pfr_availablity_status(player_ids: list): 
    """
    Update player records to indicate they are not available in PFR (unable to find HREF)

    Args:
        player_ids (list): list of player_ids to update 
    """
    query = """
        UPDATE player 
        SET pfr_available = 0
        WHERE player_id = %s
    """


    try:
        params = [(player_id,) for player_id in player_ids]

        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(
                query, params
            )  
            
            # Commit the transaction to persist data
            connection.commit()
            logging.info(
                f"Successfully updated {len(params)} player records indicating that the player is not pfr_available"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while updating player records with pfr_available status", exc_info=True
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
      INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, rush_att, rush_yds, rush_tds, tgt, rec, rec_yd, rec_td, snap_pct, off_snps)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
         INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, completions, attempts, pass_yd, pass_td, interceptions, rating, sacked, rush_att, rush_yds, rush_tds, snap_pct, off_snps)
         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
      INSERT INTO player_game_log (player_id, week, day, year, home_team, opp, result, points_for, points_allowed, tgt, rec, rec_yd, rec_td, snap_pct, off_snps)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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


def insert_player_weekly_aggregate_metrics(player_agg_metrics: list): 
    sql = f"""
            INSERT INTO player_weekly_agg_metrics (player_id, week, season, avg_fantasy_points)
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
                f"Successfully inserted {len(player_agg_metrics)} player_weekly_agg_metrics records into the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_weekly_agg_metrics records into our db: {player_agg_metrics}",
            exc_info=True,
        )
        raise e


def insert_player_depth_charts(player_depth_chart: list): 
    """
    Functionality to insert player depth chart records into our DB 

    ArgsL
        player_depth_chart (list): players depth charts to insert 
    """
    sql = """
            INSERT INTO player_depth_chart (player_id, week, season, depth_chart_pos)
            VALUES (%s, %s, %s, %s)
         """

    try:
        connection = get_connection()

        params = [
            (
                record["player_id"],
                record["week"],
                record["season"],
                record["depth_chart_pos"]
            )
            for record in player_depth_chart
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(player_depth_chart)} player_depth_chart records into the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_depth_chart records into our db: {player_depth_chart}",
            exc_info=True,
        )
        raise e


def format_and_insert_team_seasonal_general_metrics(team_stats: dict, team_conversions: dict, team_id: int, year: int):
    """
    Combine team stats and conversions into a single record and insert into our database

    Args:
        team_stats (dict): team stats
        team_conversions (dict): team conversions   
        team_id (int): team ID
        year (int): year
    """


    # fetch team home and away wins and losses 
    record = fetch_data.fetch_teams_home_away_wins_and_losses(year, team_id)
    record['team_id'] = team_id
    record['season'] = year

    # Add team stats to record
    record['total_games'] = int(record['wins']) + int(record['losses'])
    record['total_yards'] = team_stats.get('team_total_yards', 0)
    record['plays_offense'] = team_stats.get('team_plays_offense', 0)
    record['yds_per_play'] = team_stats.get('team_yds_per_play_offense', 0)
    record['turnovers'] = team_stats.get('team_turnovers', 0)
    record['first_downs'] = team_stats.get('team_first_down', 0)
    record['penalties'] = team_stats.get('team_penalties', 0)
    record['penalties_yards'] = team_stats.get('team_penalties_yds', 0)
    record['penalty_first_downs'] = team_stats.get('team_pen_fd', 0)
    record['drives'] = team_stats.get('team_drives', 0)
    record['score_pct'] = team_stats.get('team_score_pct', 0)
    record['turnover_pct'] = team_stats.get('team_turnover_pct', 0)
    record['fumble_lost'] = team_stats.get('team_fumbles_lost', 0)

    # Convert starting field position from "Own X.X" to numeric
    start_avg = team_stats.get('team_start_avg', '')
    if start_avg and 'Own ' in start_avg:
        record['start_avg'] = float(start_avg.replace('Own ', ''))
    else:
        record['start_avg'] = 0



    record['time_avg'] = convert_time_to_seconds(team_stats.get('team_time_avg', '0:00'))
    record['plays_per_drive'] = team_stats.get('team_plays_per_drive', 0)
    record['yards_per_drive'] = team_stats.get('team_yds_per_drive', 0)
    record['points_per_drive'] = team_stats.get('team_points_avg', 0)

    # Add team conversion stats to record
    record['third_down_attempts'] = int(team_conversions.get('team_third_down_att', 0))
    record['third_down_conversions'] = int(team_conversions.get('team_third_down_success', 0))
    record['third_down_pct'] = float(team_conversions.get('team_third_down_pct', '0').rstrip('%'))
    record['fourth_down_attempts'] = int(team_conversions.get('team_fourth_down_att', 0))
    record['fourth_down_conversions'] = int(team_conversions.get('team_fourth_down_success', 0))
    record['fourth_down_pct'] = float(team_conversions.get('team_fourth_down_pct', '0').rstrip('%'))
    record['red_zone_attempts'] = int(team_conversions.get('team_red_zone_att', 0))
    record['red_zone_scores'] = int(team_conversions.get('team_red_zone_scores', 0))
    record['red_zone_pct'] = float(team_conversions.get('team_red_zone_pct', '0').rstrip('%'))

    # insert record into database
    insert_team_seasonal_general_metrics(record)


def insert_team_seasonal_general_metrics(record: dict):
    """Insert teams seasonal general metrics into our database

    Args:
        record (dict): record containing relevant season team metrics
    """


    sql = """
    INSERT INTO team_seasonal_general_metrics (
        team_id, season, fumble_lost, home_wins, home_losses, away_wins, away_losses,
        wins, losses, win_pct, total_games, total_yards, plays_offense, yds_per_play,
        turnovers, first_down, penalties, penalties_yds, pen_fd, drives, score_pct,
        turnover_pct, start_avg, time_avg, plays_per_drive, yds_per_drive, points_avg,
        third_down_att, third_down_success, third_down_pct, fourth_down_att,
        fourth_down_success, fourth_down_pct, red_zone_att, red_zone_scores, red_zone_pct
    ) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        params = (
            record['team_id'], record['season'], record['fumble_lost'], 
            record['home_wins'], record['home_losses'], record['away_wins'], 
            record['away_losses'], record['wins'], record['losses'], record['win_pct'],
            record['total_games'], record['total_yards'], record['plays_offense'],
            record['yds_per_play'], record['turnovers'], record['first_downs'],
            record['penalties'], record['penalties_yards'], record['penalty_first_downs'],
            record['drives'], record['score_pct'], record['turnover_pct'],
            record['start_avg'], record['time_avg'], record['plays_per_drive'],
            record['yards_per_drive'], record['points_per_drive'],
            record['third_down_attempts'], record['third_down_conversions'],
            record['third_down_pct'], record['fourth_down_attempts'],
            record['fourth_down_conversions'], record['fourth_down_pct'],
            record['red_zone_attempts'], record['red_zone_scores'], record['red_zone_pct']
        )

        with connection.cursor() as cur:
            cur.execute(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted team_seasonal_general_metrics record for team {record['team_id']} season {record['season']}"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following team_seasonal_general_metrics record into our db: {record}",
            exc_info=True,
        )
        raise e


def insert_team_seasonal_passing_metrics(record: dict, team_id: int, season: int):
    """Insert team seasonal passing metrics into the database

    Args:
        record (dict): record containing relevant season team passing metrics
        team_id (int): team ID
        season (int): season year
    """

    sql = """
    INSERT INTO team_seasonal_passing_metrics (
        team_id, season, pass_attempts, complete_pass, incomplete_pass, passing_yards,
        pass_td, interception, net_yds_per_att, first_downs, cmp_pct, td_pct, int_pct,
        success, long, yds_per_att, adj_yds_per_att, yds_per_cmp, yds_per_g, rating,
        sacked, sacked_yds, sacked_pct, adj_net_yds_per_att, comebacks, game_winning_drives
    ) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        params = (
            team_id, season, record['team_total_pass_att'], 
            record['team_total_pass_cmp'], int(record['team_total_pass_att']) - int(record['team_total_pass_cmp']), record['team_total_pass_yds'],
            record['team_total_pass_td'], record['team_total_pass_int'], record['team_total_pass_net_yds_per_att'],
            record['team_total_pass_first_down'], float(record['team_total_pass_cmp_pct'].rstrip('%')),
            float(record['team_total_pass_td_pct'].rstrip('%')), float(record['team_total_pass_int_pct'].rstrip('%')),
            record['team_total_pass_success'], record['team_total_pass_long'],
            record['team_total_pass_yds_per_att'], record['team_total_pass_adj_yds_per_att'],
            record['team_total_pass_yds_per_cmp'], record['team_total_pass_yds_per_g'],
            record['team_total_pass_rating'], record['team_total_pass_sacked'],
            record['team_total_pass_sacked_yds'], float(record['team_total_pass_sacked_pct'].rstrip('%')),
            record['team_total_pass_adj_net_yds_per_att'], record['team_total_comebacks'],
            record['team_total_gwd']
        )

        with connection.cursor() as cur:
            cur.execute(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted team_seasonal_passing_metrics record for team {team_id} season {season}"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following team_seasonal_passing_metrics record into our db: {record}",
            exc_info=True,
        )
        raise e


def insert_team_seasonal_rushing_metrics(df: pd.DataFrame, teams: dict):
    """Insert teams seasonal rushing metrics into our database

    Args:
        df (pd.DataFrame): dataframe containing relevant season team metrics
        teams (dict): mapping of a player
    """

    sql = """
    INSERT INTO team_seasonal_rushing_metrics (
        team_id, season, rushing_yards, rush_td
    ) 
    VALUES (%s, %s, %s, %s, %s)
    """


    try:
        connection = get_connection()

        params = [
            (
                team_service.get_team_id_by_name(
                    next(team["team"] for team in teams if team["acronym"] == row["team"])
                ),  # transform acronym into corresponding ID
                row["season"],
                row["rushing_yards"],
                row["run_fumble"]
            )
            for _, row in df.iterrows()
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(df)} team_seasonal_rushing_metrics records into the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following team_seasonal_rushing_metrics records into our db: {params}",
            exc_info=True,
        )
        raise e


def insert_player_demographics(df: pd.DataFrame):
    """Insert player demographics (i.e age, height, weight) for a given season

    Args:
        df (pd.DataFrame): dataframe containing relevant player season metrics
    """

    sql = """
    INSERT INTO player_demographics (
        player_id, season, age, height, weight
    ) 
    VALUES (%s, %s, %s, %s, %s)
    """

    params = []
    ids = set()  # Track unique player IDs

    try:
        connection = get_connection()

        for _, row in df.iterrows(): 
            try:
                # fetch player ID by cleaned name
                player_id = player_service.get_player_id_by_normalized_name(row["player_name"])

                # skip if player ID has already been processed
                if player_id not in ids:
                    ids.add(player_id)
                else:
                    logging.info(f"Player ID {player_id} already persisted")
                    continue

                # append data to parameters for insertion
                params.append((
                    player_id,
                    row["season"],
                    row["age"],
                    row["height"],
                    row["weight"]
                ))
            except Exception as e:
                logging.error(f"The following error occurred while generating tuple for name: {row['player_name']}", exc_info=True)
                continue
                
        # filter out existing entries 
        filtered_params = [] 
        for param in params: 
            if fetch_data.retrieve_player_demographics_record_by_pk(param[1], param[0]) is None:
                filtered_params.append(param)

        with connection.cursor() as cur:
            cur.executemany(sql, filtered_params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(filtered_params)} player_demographic records into the database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_demographics records into our db: {filtered_params}",
            exc_info=True,
        )
        raise 

def insert_team_seasonal_rushing_and_receiving_metrics(record: dict, team_id: int, season: int):
    """Insert team seasonal rushing and receiving metrics into our database

    Args:
        record (dict): record containing relevant season team rushing and receiving metrics
        team_id (int): team ID
        season (int): season year
    """

    sql = """
    INSERT INTO team_seasonal_rushing_receiving_metrics (
        team_id, season, rush_att, rush_yds_per_att, rush_fd, rush_success, rush_long,
        rush_yds_per_g, rush_att_per_g, rush_yds, rush_tds, targets, rec, rec_yds,
        rec_yds_per_rec, rec_td, rec_first_down, rec_success, rec_long, rec_per_g,
        rec_yds_per_g, catch_pct, rec_yds_per_tgt, touches, yds_per_touch,
        yds_from_scrimmage, rush_receive_td, fumbles
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        params = (
            team_id, season,
            int(record['team_total_rush_att']), float(float(record['team_total_rush_yds'])/int(record['team_total_rush_att'])),
            int(record['team_total_rush_first_down']), float(record['team_total_rush_success']),
            int(record['team_total_rush_long']), float(record['team_total_rush_yds_per_g']),
            float(record['team_total_rush_att_per_g']), int(record['team_total_rush_yds']),
            int(record['team_total_rush_td']), int(record['team_total_targets']),
            int(record['team_total_rec']), int(record['team_total_rec_yds']),
            float(record['team_total_rec_yds_per_rec']), int(record['team_total_rec_td']),
            int(record['team_total_rec_first_down']), float(record['team_total_rec_success']),
            int(record['team_total_rec_long']), float(record['team_total_rec_per_g']),
            float(record['team_total_rec_yds_per_g']), float(record['team_total_catch_pct']),
            float(record['team_total_rec_yds_per_tgt']), int(record['team_total_touches']),
            float(record['team_total_yds_per_touch']), float(record['team_total_yds_from_scrimmage']),
            int(record['team_total_rush_receive_td']), int(record['team_total_fumbles'])
        )

        with connection.cursor() as cur:
            cur.execute(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted team_seasonal_rushing_receiving_metrics record for team {team_id} season {season}"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following team_seasonal_rushing_receiving_metrics record into our db: {record}",
            exc_info=True,
        )
        raise e

    
def insert_team_seasonal_kicking_and_punting_metrics(punting_record: dict, kicking_record: dict, team_id: int, season: int):
    """Insert team seasonal kicking and punting metrics into our database

    Args:
        punting_record (dict): record containing relevant season team punting metrics
        kicking_record (dict): record containing relevant season team kicking metrics
        team_id (int): team ID
        season (int): season year
    """
    # Insert punting metrics
    punting_sql = """
    INSERT INTO team_seasonal_punting_metrics (
        team_id, season, team_total_punt, team_total_punt_yds, team_total_punt_yds_per_punt,
        team_total_punt_ret_yds_opp, team_total_punt_net_yds, team_total_punt_net_yds_per_punt,
        team_total_punt_long, team_total_punt_tb, team_total_punt_tb_pct,
        team_total_punt_in_20, team_total_punt_in_20_pct, team_total_punt_blocked
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    punting_params = (
        team_id, season,
        int(punting_record['team_total_punt']),
        int(punting_record['team_total_punt_yds']),
        float(punting_record['team_total_punt_yds_per_punt']),
        int(punting_record['team_total_punt_ret_yds_opp']),
        int(punting_record['team_total_punt_net_yds']),
        float(punting_record['team_total_punt_net_yds_per_punt']),
        int(punting_record.get('team_total_punt_long', 0)),
        int(punting_record['team_total_punt_tb']),
        float(punting_record['team_total_punt_tb_pct']),
        int(punting_record['team_total_punt_in_20']),
        float(punting_record['team_total_punt_in_20_pct']),
        int(punting_record['team_total_punt_blocked'])
    )

    # Insert kicking metrics
    kicking_sql = """
    INSERT INTO team_seasonal_kicking_metrics (
        team_id, season, team_total_fga1, team_total_fgm1, team_total_fga2,
        team_total_fgm2, team_total_fga3, team_total_fgm3, team_total_fga4,
        team_total_fgm4, team_total_fga5, team_total_fgm5, team_total_fga,
        team_total_fgm, team_total_fg_long, team_total_fg_pct, team_total_xpa,
        team_total_xpm, team_total_xp_pct, team_total_kickoff, team_total_kickoff_yds,
        team_total_kickoff_tb, team_total_kickoff_tb_pct, team_total_kickoff_yds_avg
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s)
    """

    kicking_params = (
        team_id, season,
        int(kicking_record['team_total_fga1']),
        int(kicking_record['team_total_fgm1']),
        int(kicking_record['team_total_fga2']),
        int(kicking_record['team_total_fgm2']), 
        int(kicking_record['team_total_fga3']),
        int(kicking_record['team_total_fgm3']),
        int(kicking_record['team_total_fga4']),
        int(kicking_record['team_total_fgm4']),
        int(kicking_record['team_total_fga5']),
        int(kicking_record.get('team_total_fgm5', 0)),
        int(kicking_record['team_total_fga']),
        int(kicking_record['team_total_fgm']),
        int(kicking_record['team_total_fg_long']),
        float(kicking_record['team_total_fg_pct']),
        int(kicking_record['team_total_xpa']),
        int(kicking_record['team_total_xpm']),
        float(kicking_record['team_total_xp_pct']),
        int(kicking_record['team_total_kickoff']),
        int(kicking_record['team_total_kickoff_yds']),
        int(kicking_record.get('team_total_kickoff_tb', 0)),
        float(kicking_record['team_total_kickoff_tb_pct']),
        float(kicking_record['team_total_kickoff_yds_avg'])
    )

    try:
        connection = get_connection()
        with connection.cursor() as cur:
            # Insert punting metrics
            cur.execute(punting_sql, punting_params)
            logging.info(f"Successfully inserted team_seasonal_punting_metrics record for team {team_id} season {season}")

            # Insert kicking metrics  
            cur.execute(kicking_sql, kicking_params)
            logging.info(f"Successfully inserted team_seasonal_kicking_metrics record for team {team_id} season {season}")

            connection.commit()

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting kicking/punting metrics for team {team_id} season {season}",
            exc_info=True,
        )
        raise e


def insert_team_seasonal_defense_and_fumbles_metrics(team_stats: dict, team_defensive_stats: dict, team_conversions: dict, team_id: int, season: int):
    """Insert team seasonal defense and fumbles metrics into our database

    Args:
        team_stats (dict): Dictionary containing defensive stats
        team_defensive_stats (dict): Dictionary containing additional defensive stats
        team_conversions (dict): Dictionary containing defensive conversion stats
        team_id (int): Team ID
        season (int): Season year
    """

    # Create record dictionary
    record = {
        'team_id': team_id,
        'season': season,
        'points': team_stats.get('opp_points', 0),
        'total_yards': team_stats.get('opp_total_yards', 0),
        'plays_offense': team_stats.get('opp_plays_offense', 0),
        'yds_per_play_offense': team_stats.get('opp_yds_per_play_offense', 0),
        'turnovers': team_stats.get('opp_turnovers', 0),
        'fumbles_lost': team_stats.get('opp_fumbles_lost', 0),
        'first_down': team_stats.get('opp_first_down', 0),
        'pass_cmp': team_stats.get('opp_pass_cmp', 0),
        'pass_att': team_stats.get('opp_pass_att', 0),
        'pass_yds': team_stats.get('opp_pass_yds', 0),
        'pass_td': team_stats.get('opp_pass_td', 0),
        'pass_int': team_stats.get('opp_pass_int', 0),
        'pass_net_yds_per_att': team_stats.get('opp_pass_net_yds_per_att', 0),
        'pass_fd': team_stats.get('opp_pass_fd', 0),
        'rush_att': team_stats.get('opp_rush_att', 0),
        'rush_yds': team_stats.get('opp_rush_yds', 0),
        'rush_td': team_stats.get('opp_rush_td', 0),
        'rush_yds_per_att': team_stats.get('opp_rush_yds_per_att', 0),
        'rush_fd': team_stats.get('opp_rush_fd', 0),
        'penalties': team_stats.get('opp_penalties', 0),
        'penalties_yds': team_stats.get('opp_penalties_yds', 0),
        'pen_fd': team_stats.get('opp_pen_fd', 0),
        'drives': team_stats.get('opp_drives', 0),
        'score_pct': team_stats.get('opp_score_pct', 0),
        'turnover_pct': team_stats.get('opp_turnover_pct', 0),
        'plays_per_drive': team_stats.get('opp_plays_per_drive', 0),
        'yds_per_drive': team_stats.get('opp_yds_per_drive', 0),
        'points_avg': team_stats.get('opp_points_avg', 0),
        'time_avg': team_stats.get('opp_time_avg', '0:00'),
        # Additional defensive stats
        'def_int': int(team_defensive_stats.get('team_total_def_int', 0)),
        'def_int_yds': int(team_defensive_stats.get('team_total_def_int_yds', 0)),
        'def_int_td': int(team_defensive_stats.get('team_total_def_int_td', 0)),
        'def_int_long': int(team_defensive_stats.get('team_total_def_int_long', 0)),
        'pass_defended': int(team_defensive_stats.get('team_total_pass_defended', 0)),
        'fumbles_forced': int(team_defensive_stats.get('team_total_fumbles_forced', 0)),
        'fumbles_rec': int(team_defensive_stats.get('team_total_fumbles_rec', 0)),
        'fumbles_rec_yds': int(team_defensive_stats.get('team_total_fumbles_rec_yds', 0)),
        'fumbles_rec_td': int(team_defensive_stats.get('team_total_fumbles_rec_td', 0)),
        'sacks': float(team_defensive_stats.get('team_total_sacks', 0)),
        'tackles_combined': int(team_defensive_stats.get('team_total_tackles_combined', 0)),
        'tackles_solo': int(team_defensive_stats.get('team_total_tackles_solo', 0)),
        'tackles_assists': int(team_defensive_stats.get('team_total_tackles_assists', 0)),
        'tackles_loss': int(team_defensive_stats.get('team_total_tackles_loss', 0)),
        'qb_hits': int(team_defensive_stats.get('team_total_qb_hits', 0)),
        'safety_md': int(team_defensive_stats.get('team_total_safety_md', 0))
    }

    # Convert starting field position from "Own X.X" to numeric
    start_avg = team_stats.get('opp_start_avg', '')
    if start_avg and 'Own ' in start_avg:
        record['start_avg'] = float(start_avg.replace('Own ', ''))
    else:
        record['start_avg'] = 0

    # Add conversion stats
    record['third_down_att'] = int(team_conversions.get('opp_third_down_att', 0))
    record['third_down_success'] = int(team_conversions.get('opp_third_down_success', 0))
    record['third_down_pct'] = float(team_conversions.get('opp_third_down_pct', '0').rstrip('%'))
    record['fourth_down_att'] = int(team_conversions.get('opp_fourth_down_att', 0))
    record['fourth_down_success'] = int(team_conversions.get('opp_fourth_down_success', 0))
    record['fourth_down_pct'] = float(team_conversions.get('opp_fourth_down_pct', '0').rstrip('%'))
    record['red_zone_att'] = int(team_conversions.get('opp_red_zone_att', 0))
    record['red_zone_scores'] = int(team_conversions.get('opp_red_zone_scores', 0))
    record['red_zone_pct'] = float(team_conversions.get('opp_red_zone_pct', '0').rstrip('%'))

    sql = """
    INSERT INTO team_seasonal_defensive_metrics (
        team_id, season, points, total_yards, plays_offense, yds_per_play_offense,
        turnovers, fumbles_lost, first_down, pass_cmp, pass_att, pass_yds, pass_td,
        pass_int, pass_net_yds_per_att, pass_fd, rush_att, rush_yds, rush_td,
        rush_yds_per_att, rush_fd, penalties, penalties_yds, pen_fd, drives,
        score_pct, turnover_pct, start_avg, time_avg, plays_per_drive,
        yds_per_drive, points_avg, third_down_att, third_down_success,
        third_down_pct, fourth_down_att, fourth_down_success, fourth_down_pct,
        red_zone_att, red_zone_scores, red_zone_pct, def_int, def_int_yds,
        def_int_td, def_int_long, pass_defended, fumbles_forced, fumbles_rec,
        fumbles_rec_yds, fumbles_rec_td, sacks, tackles_combined, tackles_solo,
        tackles_assists, tackles_loss, qb_hits, safety_md
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s
    )
    """

    params = (
        record['team_id'], record['season'], record['points'], record['total_yards'],
        record['plays_offense'], record['yds_per_play_offense'], record['turnovers'],
        record['fumbles_lost'], record['first_down'], record['pass_cmp'],
        record['pass_att'], record['pass_yds'], record['pass_td'], record['pass_int'],
        record['pass_net_yds_per_att'], record['pass_fd'], record['rush_att'],
        record['rush_yds'], record['rush_td'], record['rush_yds_per_att'],
        record['rush_fd'], record['penalties'], record['penalties_yds'],
        record['pen_fd'], record['drives'], record['score_pct'],
        record['turnover_pct'], record['start_avg'], convert_time_to_seconds(record['time_avg']),
        record['plays_per_drive'], record['yds_per_drive'], record['points_avg'],
        record['third_down_att'], record['third_down_success'], record['third_down_pct'],
        record['fourth_down_att'], record['fourth_down_success'], record['fourth_down_pct'],
        record['red_zone_att'], record['red_zone_scores'], record['red_zone_pct'],
        record['def_int'], record['def_int_yds'], record['def_int_td'],
        record['def_int_long'], record['pass_defended'], record['fumbles_forced'],
        record['fumbles_rec'], record['fumbles_rec_yds'], record['fumbles_rec_td'],
        record['sacks'], record['tackles_combined'], record['tackles_solo'],
        record['tackles_assists'], record['tackles_loss'], record['qb_hits'],
        record['safety_md']
    )

    try:
        connection = get_connection()
        with connection.cursor() as cur:
            cur.execute(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted team_seasonal_defensive_metrics record for team {team_id} season {season}"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following team_seasonal_defensive_metrics record into our db: {record}",
            exc_info=True,
        )
        raise e



def insert_team_seasonal_scoring_metrics(record: dict, team_id: int, season: int):
    """Insert team seasonal scoring summary metrics into our database

    Args:
        record (dict): record containing relevant season team scoring metrics
    """
    sql = """
    INSERT INTO team_seasonal_scoring_metrics (
        team_id, season, rush_td, rec_td, punt_ret_td, kick_ret_td, fumbles_rec_td,
        def_int_td, other_td, total_td, two_pt_md, def_two_pt, xpm, xpa, fgm, fga,
        safety_md, scoring
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        params = (
            team_id, season,
            int(record['team_total_rush_td']), int(record['team_total_rec_td']),
            int(record['team_total_punt_ret_td']), int(record['team_total_kick_ret_td']),
            int(record['team_total_fumbles_rec_td']), int(record['team_total_def_int_td']),
            int(record['team_total_other_td']), int(record['team_total_total_td']),
            int(record['team_total_two_pt_md']), int(record.get('team_total_def_two_pt', 0)),
            int(record['team_total_xpm']), int(record['team_total_xpa']),
            int(record['team_total_fgm']), int(record['team_total_fga']),
            int(record['team_total_safety_md']), int(record['team_total_scoring'])
        )

        with connection.cursor() as cur:
            cur.execute(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted team_seasonal_scoring_metrics record for team {team_id} season {season}"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following team_seasonal_scoring_metrics record into our db: {record}",
            exc_info=True,
        )
        raise e


def insert_team_seasonal_rankings_metrics(team_stats: dict, team_conversions: dict, team_id: int, season: int):
    """Insert team seasonal rankings metrics into our database

    Args:
        team_stats (dict): Dictionary containing team stats with offensive and defensive rankings
        team_conversions (dict): Dictionary containing team conversion rankings
        team_id (int): Team ID
        season (int): Season year
    """
    sql = """
    INSERT INTO team_seasonal_ranks (
        team_id, season,
        def_points, def_total_yards, def_turnovers, def_fumbles_lost,
        def_first_down, def_pass_att, def_pass_yds, def_pass_td, def_pass_int,
        def_pass_net_yds_per_att, def_rush_att, def_rush_yds, def_rush_td,
        def_rush_yds_per_att, def_score_pct, def_turnover_pct, def_start_avg,
        def_time_avg, def_plays_per_drive, def_yds_per_drive, def_points_avg,
        def_third_down_pct, def_fourth_down_pct, def_red_zone_pct,
        off_points, off_total_yards, off_turnovers, off_fumbles_lost,
        off_first_down, off_pass_att, off_pass_yds, off_pass_td, off_pass_int,
        off_pass_net_yds_per_att, off_rush_att, off_rush_yds, off_rush_td,
        off_rush_yds_per_att, off_score_pct, off_turnover_pct, off_start_avg,
        off_time_avg, off_plays_per_drive, off_yds_per_drive, off_points_avg,
        off_third_down_pct, off_fourth_down_pct, off_red_zone_pct
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        params = (
            team_id, season,
            int(team_stats['def_rank_points']),
            int(team_stats['def_rank_total_yards']),
            int(team_stats['def_rank_turnovers']),
            int(team_stats['def_rank_fumbles_lost']),
            int(team_stats['def_rank_first_down']),
            int(team_stats['def_rank_pass_att']),
            int(team_stats['def_rank_pass_yds']),
            int(team_stats['def_rank_pass_td']),
            int(team_stats['def_rank_pass_int']),
            float(team_stats['def_rank_pass_net_yds_per_att']),
            int(team_stats['def_rank_rush_att']),
            int(team_stats['def_rank_rush_yds']),
            int(team_stats['def_rank_rush_td']),
            float(team_stats['def_rank_rush_yds_per_att']),
            float(team_stats['def_rank_score_pct']),
            float(team_stats['def_rank_turnover_pct']),
            float(team_stats['def_rank_start_avg']),
            float(team_stats['def_rank_time_avg']),
            float(team_stats['def_rank_plays_per_drive']),
            float(team_stats['def_rank_yds_per_drive']),
            float(team_stats['def_rank_points_avg']),
            float(team_conversions.get('def_rank_third_down_pct', 0)),
            float(team_conversions.get('def_rank_fourth_down_pct', 0)),
            float(team_conversions.get('def_rank_red_zone_pct', 0)),
            int(team_stats['off_rank_points']),
            int(team_stats['off_rank_total_yards']),
            int(team_stats['off_rank_turnovers']),
            int(team_stats['off_rank_fumbles_lost']),
            int(team_stats['off_rank_first_down']),
            int(team_stats['off_rank_pass_att']),
            int(team_stats['off_rank_pass_yds']),
            int(team_stats['off_rank_pass_td']),
            int(team_stats['off_rank_pass_int']),
            float(team_stats['off_rank_pass_net_yds_per_att']),
            int(team_stats['off_rank_rush_att']),
            int(team_stats['off_rank_rush_yds']),
            int(team_stats['off_rank_rush_td']),
            float(team_stats['off_rank_rush_yds_per_att']),
            float(team_stats['off_rank_score_pct']),
            float(team_stats['off_rank_turnover_pct']),
            float(team_stats['off_rank_start_avg']),
            float(team_stats['off_rank_time_avg']),
            float(team_stats['off_rank_plays_per_drive']),
            float(team_stats['off_rank_yds_per_drive']),
            float(team_stats['off_rank_points_avg']),
            float(team_conversions.get('off_rank_third_down_pct', 0)),
            float(team_conversions.get('off_rank_fourth_down_pct', 0)), 
            float(team_conversions.get('off_rank_red_zone_pct', 0))
        )

        with connection.cursor() as cur:
            cur.execute(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted team_seasonal_ranks record for team {team_id} season {season}"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following team_seasonal_ranks record into our db: {team_stats}",
            exc_info=True,
        )
        raise e


def insert_player_seasonal_passing_metrics(record: dict, year: int, team_id: int):
    """Insert player seasonal passing metrics into our database

    Args:
        record (dict): dictionary containing player passing metrics for a season
        year (int): season year
        team_id (int): team id
    """
    try:
        sql = """
            INSERT INTO player_seasonal_passing_metrics (
                player_id, team_id, season, games_started, qb_rec, pass_att, pass_cmp_pct,
                pass_yds, pass_td, pass_td_pct, pass_int, pass_int_pct,
                pass_first_down, pass_success, pass_long, pass_yds_per_att,
                pass_adj_yds_per_att, pass_yds_per_cmp, pass_yds_per_g,
                pass_rating, qbr, pass_sacked, pass_sacked_yds, pass_sacked_pct,
                pass_net_yds_per_att, pass_adj_net_yds_per_att, comebacks,
                game_winning_drives
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """

        for player_name, stats in record.items():
            player_id = player_service.get_player_id_by_normalized_name(player_name)
            
            if player_id is None:
                logging.info(f"Skipping insert for player {player_name} - no matching player_id found")
                continue

            params = (
                player_id,
                team_id,
                year,
                int(stats.get('games_started', 0)),
                stats.get('qb_rec', '0-0-0'),
                int(stats.get('pass_att', 0)),
                float(stats.get('pass_cmp_pct', 0)),
                int(stats.get('pass_yds', 0)),
                int(stats.get('pass_td', 0)), 
                float(stats.get('pass_td_pct', 0)),
                int(stats.get('pass_int', 0)),
                float(stats.get('pass_int_pct', 0)),
                int(stats.get('pass_first_down', 0)),
                float(stats.get('pass_success', 0)),
                int(stats.get('pass_long', 0)),
                float(stats.get('pass_yds_per_att', 0)),
                float(stats.get('pass_adj_yds_per_att', 0)),
                float(stats.get('pass_yds_per_cmp', 0)),
                float(stats.get('pass_yds_per_g', 0)),
                float(stats.get('pass_rating', 0)),
                float(stats.get('qbr', 0)),
                int(stats.get('pass_sacked', 0)),
                int(stats.get('pass_sacked_yds', 0)),
                float(stats.get('pass_sacked_pct', 0)),
                float(stats.get('pass_net_yds_per_att', 0)),
                float(stats.get('pass_adj_net_yds_per_att', 0)),
                int(stats.get('comebacks', 0)),
                int(stats.get('gwd', 0))
            )

            try:
                connection = get_connection()
                with connection.cursor() as cur:
                    cur.execute(sql, params)
                    connection.commit()
                    logging.info(f"Successfully inserted player_seasonal_passing_metrics record for player {player_name} season {year}")
            except Exception as e:
                logging.error(f"Database connection error: {e}")
                raise e


    except Exception as e:
        logging.error(
            f"An exception occurred while inserting player_seasonal_passing_metrics records into db: {record}",
            exc_info=True
        )
        raise e


def insert_player_seasonal_rushing_and_receiving_metrics(record: dict, year: int, team_id: int):
    """Insert player seasonal rushing and receiving metrics into our database

    Args:
        record (dict): dictionary containing player rushing and receiving metrics for a season
        year (int): season year
        team_id (int): team id
    """
    sql = """
        INSERT INTO player_seasonal_rushing_receiving_metrics (
            player_id, team_id, season, games_started, rush_att, rush_yds_per_att, rush_fd,
            rush_success, rush_long, rush_yds_per_g, rush_att_per_g, rush_yds, rush_tds,
            targets, rec, rec_yds, rec_yds_per_rec, rec_td, rec_first_down, rec_success,
            rec_long, rec_per_g, rec_yds_per_g, catch_pct, rec_yds_per_tgt, touches,
            yds_per_touch, yds_from_scrimmage, rush_receive_td, fumbles
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """

    try:
        for player_name, stats in record.items():
            # Get player ID from name
            player_id = player_service.get_player_id_by_normalized_name(player_name)

            if player_id is None:
                logging.info(f"Skipping insert for player {player_name} - no matching player_id found")
                continue

            params = (
                player_id,
                team_id,
                year,
                int(stats.get('games_started', 0)),
                int(stats.get('rush_att', 0)),
                float(stats.get('rush_yds_per_att', 0)),
                int(stats.get('rush_fd', 0)),
                float(stats.get('rush_success', 0)),
                int(stats.get('rush_long', 0)),
                float(stats.get('rush_yds_per_g', 0)),
                float(stats.get('rush_att_per_g', 0)),
                int(stats.get('rush_yds', 0)),
                int(stats.get('rush_tds', 0)),
                int(stats.get('targets', 0)),
                int(stats.get('rec', 0)),
                int(stats.get('rec_yds', 0)),
                float(stats.get('rec_yds_per_rec', 0)),
                int(stats.get('rec_td', 0)),
                int(stats.get('rec_first_down', 0)),
                float(stats.get('rec_success', 0)),
                int(stats.get('rec_long', 0)),
                float(stats.get('rec_per_g', 0)),
                float(stats.get('rec_yds_per_g', 0)),
                float(stats.get('catch_pct', 0)),
                float(stats.get('rec_yds_per_tgt', 0)),
                int(stats.get('touches', 0)),
                float(stats.get('yds_per_touch', 0)),
                float(stats.get('yds_from_scrimmage', 0)),
                int(stats.get('rush_receive_td', 0)),
                int(stats.get('fumbles', 0))
            )

            try:
                connection = get_connection()
                with connection.cursor() as cur:
                    cur.execute(sql, params)
                    connection.commit()
                    logging.info(f"Successfully inserted player_seasonal_rushing_receiving_metrics record for player {player_name} season {year}")
            except Exception as e:
                logging.error(f"Database connection error: {e}")
                raise e

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting player_seasonal_rushing_receiving_metrics records into db: {record}",
            exc_info=True
        )
        raise e
def insert_player_seasonal_scoring_metrics(record: dict, year: int, team_id: int):
    """Insert player seasonal scoring metrics into our database

    Args:
        record (dict): dictionary containing player scoring metrics for a season
        year (int): season year
        team_id (int): team id
    """
    try:
        sql = """
            INSERT INTO player_seasonal_scoring_metrics (
                player_id, team_id, season, rush_td, rec_td, punt_ret_td, kick_ret_td,
                fumbles_rec_td, def_int_td, other_td, total_td, two_pt_md,
                def_two_pt, xpm, xpa, fgm, fga, safety_md, scoring
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """

        for player_name, stats in record.items():
            # Get player ID from player service
            player_id = player_service.get_player_id_by_normalized_name(player_name)
            
            if player_id is None:
                logging.warning(f"Could not find player_id for {player_name}, skipping record")
                continue

            params = (
                player_id,
                team_id,
                year,
                int(stats.get('rush_td', 0)),
                int(stats.get('rec_td', 0)), 
                int(stats.get('punt_ret_td', 0)),
                int(stats.get('kick_ret_td', 0)),
                int(stats.get('fumbles_rec_td', 0)),
                int(stats.get('def_int_td', 0)),
                int(stats.get('other_td', 0)),
                int(stats.get('total_td', 0)),
                int(stats.get('two_pt_md', 0)),
                int(stats.get('def_two_pt', 0)),
                int(stats.get('xpm', 0)),
                int(stats.get('xpa', 0)),
                int(stats.get('fgm', 0)),
                int(stats.get('fga', 0)),
                int(stats.get('safety_md', 0)),
                int(stats.get('scoring', 0))
            )

            try:
                connection = get_connection()
                with connection.cursor() as cur:
                    cur.execute(sql, params)
                    connection.commit()
                    logging.info(f"Successfully inserted player_seasonal_scoring_metrics record for player {player_name} season {year}")
            except Exception as e:
                logging.error(f"Database connection error: {e}")
                raise e

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting player_seasonal_scoring_metrics records into db: {record}",
            exc_info=True
        )
        raise e

def insert_player_advanced_passing_metrics(records: list, player_id: int, season: int):
    """Insert player advanced passing metrics into our database

    Args:
        records (list): list of records containing relevant advanced passing metrics
        player_id (int): ID of the player
        season (int): season year
    """
    sql = """
    INSERT INTO player_advanced_passing (
        player_id, week, season, age, first_downs, first_down_passing_per_pass_play,
        intended_air_yards, intended_air_yards_per_pass_attempt, completed_air_yards,
        completed_air_yards_per_cmp, completed_air_yards_per_att, yds_after_catch,
        yds_after_catch_per_cmp, drops, drop_pct, poor_throws, poor_throws_pct, sacked,
        blitzed, hurried, hits, pressured, pressured_pct, scrmbl, yds_per_scrmbl
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        params = [
            (
                player_id, int(record['week']), season,
                float(record.get('age', -1.0)),
                int(record.get('first_downs', -1)),
                float(record.get('first_down_passing_per_pass_play', -1)),
                float(record.get('intended_air_yards', -1)),
                float(record.get('intended_air_yards_per_pass_attempt', -1)),
                float(record.get('completed_air_yards', -1)),
                float(record.get('completed_air_yards_per_cmp', -1)),
                float(record.get('completed_air_yards_per_att', -1)),
                float(record.get('yds_after_catch', -1)),
                float(record.get('yds_after_catch_per_cmp', -1)),
                int(record.get('drops', -1)),
                float(record.get('drop_pct', -1)),
                int(record.get('poor_throws', -1)),
                float(record.get('poor_throws_pct', -1)),
                int(record.get('sacked', -1)),
                int(record.get('blitzed', -1)),
                int(record.get('hurried', -1)),
                int(record.get('hits', -1)),
                int(record.get('pressured', -1)),
                float(record.get('pressured_pct', -1)),
                int(record.get('scrmbl', -1)),
                float(record.get('yds_per_scrmbl', -1))
            )
        for record in records]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted player_advanced_passing records for player {player_id} and season {season}"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_advanced_passing record into our db: {records}",
            exc_info=True,
        )
        raise e


def insert_player_advanced_rushing_receiving_metrics(records: list, player_id: int, season: int):
    """Insert player advanced rushing and receiving metrics into our database

    Args:
        records (list): list of records containing relevant advanced rushing and receiving metrics
        player_id (int): ID of the player
        season (int): season year
    """
    sql = """
    INSERT INTO player_advanced_rushing_receiving (
        player_id, week, season, age, rush_first_downs, rush_yds_before_contact,
        rush_yds_before_contact_per_att, rush_yds_afer_contact, rush_yds_after_contact_per_att,
        rush_brkn_tackles, rush_att_per_brkn_tackle, rec_first_downs, yds_before_catch,
        yds_before_catch_per_rec, yds_after_catch, yds_after_catch_per_rec, avg_depth_of_tgt,
        rec_brkn_tackles, rec_per_brkn_tackle, dropped_passes, drop_pct, int_when_tgted, qbr_when_tgted
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        params = [
        (
            player_id, int(record.get('week', -1)), season,
            float(record.get('age', -1)),
            int(record.get('rush_first_downs', -1)),
            float(record.get('rush_yds_before_contact', -1)),
            float(record.get('rush_yds_before_contact_per_att', -1)),
            float(record.get('rush_yds_after_contact', -1)),
            float(record.get('rush_yds_after_contact_per_att', -1)),
            int(record.get('rush_brkn_tackles', -1)),
            float(record.get('rush_att_per_brkn_tackle', -1)),
            int(record.get('rec_first_downs', -1)),
            float(record.get('yds_before_catch', -1)),
            float(record.get('yds_before_catch_per_rec', -1)),
            float(record.get('yds_after_catch', -1)),
            float(record.get('yds_after_catch_per_rec', -1)),
            float(record.get('avg_depth_of_tgt', -1)),
            int(record.get('rec_brkn_tackles', -1)),
            float(record.get('rec_per_brkn_tackle', -1)),
            int(record.get('dropped_passes', -1)),
            float(record.get('drop_pct', -1)),
            int(record.get('int_when_tgted', -1)),
            float(record.get('qbr_when_tgted', -1))
        )
        for record in records ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted player_advanced_rushing_receiving records for player {player_id} and season {season}"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_advanced_rushing_receiving records into our db: {records}",
            exc_info=True,
        )
        raise e


def convert_time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(":"))
    return float(minutes * 60 + seconds)