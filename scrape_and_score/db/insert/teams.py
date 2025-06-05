from db.connection import get_connection
import logging
import pandas as pd
from datetime import datetime
from db.read.teams import fetch_teams_home_away_wins_and_losses
from db.insert.utils import convert_time_to_seconds
from service import team_service


def insert_teams(teams: list):
    """
    Functionality to persist mutliple teams into our db 

    Args: 
        teams (list): list of teams to insert into our db 
    Returns: 
        teams (list): mapping of team names and ids we inserted

    """

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


def insert_team(team_name: str):
    """
    Functionality to persist a single team into our db 

    Args: 
        team_name (str): team to insert into our db 

    Returns: 
        team_id (int): id corresponding to a particular team 
    """

    sql = "INSERT INTO team (name) VALUES (%s) RETURNING team_id"

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (team_name,))  

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
    

def insert_team_game_logs(game_logs: list):
    """
    Functionality to persist multiple game logs for a team 

    Args:
        game_logs (list): list of tuples to insert into team_game_logs 
    
    Returns:
        None
    """

    sql = """
    INSERT INTO team_game_log (
        team_id, week, day, year, rest_days, home_team, distance_traveled, opp, result,
        points_for, points_allowed, tot_yds, pass_yds, rush_yds, pass_tds, pass_cmp,
        pass_att, pass_cmp_pct, rush_att, rush_tds, yds_gained_per_pass_att,
        adj_yds_gained_per_pass_att, pass_rate, sacked, sack_yds_lost, rush_yds_per_att,
        total_off_plays, yds_per_play, fga, fgm, xpa, xpm, total_punts,
        punt_yds, pass_fds, rsh_fds, pen_fds, total_fds, thrd_down_conv, thrd_down_att,
        fourth_down_conv, fourth_down_att, penalties, penalty_yds, fmbl_lost,
        int, turnovers, time_of_poss
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s
    )

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
    


def update_team_game_log_game_date(game_date: datetime, pk: dict):
    """
    Update team game log record with up-to-date game date 

    Args:
        game_date (datetime): new datetime to assign to team game log record
        pk (dict): PK of team game log to update
    """

    sql = """
        UPDATE team_game_log
        SET
            game_date = %s
        WHERE team_id = %s AND week = %s AND year = %s
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (game_date, pk['team_id'], pk['week'], pk['year']))
            connection.commit()
            logging.info(f"Successfully updated team game log's (PK={pk}) game_date in the database")

    except Exception as e:
        logging.error(
            f"An exception occurred while updating the team game log game_date: {pk}",
            exc_info=True,
        )
        raise e


def insert_upcoming_team_game_logs(records: list):
    """
    Functionality to insert upcoming team game logs into our database 

    Args:
        records (list): the records to persist 
    """
    
    sql = """
      INSERT INTO team_game_log (team_id, week, year, rest_days, home_team, distance_traveled, opp, game_date, day)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) 
    """

    try:
        connection = get_connection()

        params = [
            (record["team_id"], record["week"], record["year"], record["rest_days"], record['is_home'], record['distance_traveled'], record['opp'], record['game_date'], record['day'])
            for record in records 
        ]  


        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(f"Successfully inserted {len(records)} team game logs for upcoming NFL games in the database")

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the team game logs: {records}",
            exc_info=True,
        )
        raise e



def update_team_game_logs(game_logs: list):
    """
    Functionality to update multiple team game logs.

    Args:
        game_logs (list): list of tuples where each tuple contains all updatable fields followed by team_id, week, and year.
    """

    sql = """
    UPDATE team_game_log
    SET
        day = %s,
        rest_days = %s,
        home_team = %s,
        distance_traveled = %s,
        opp = %s,
        result = %s,
        points_for = %s,
        points_allowed = %s,
        tot_yds = %s,
        pass_yds = %s,
        rush_yds = %s,
        pass_tds = %s,
        pass_cmp = %s,
        pass_att = %s,
        pass_cmp_pct = %s,
        rush_att = %s,
        rush_tds = %s,
        yds_gained_per_pass_att = %s,
        adj_yds_gained_per_pass_att = %s,
        pass_rate = %s,
        sacked = %s,
        sack_yds_lost = %s,
        rush_yds_per_att = %s,
        total_off_plays = %s,
        yds_per_play = %s,
        fga = %s,
        fgm = %s,
        xpa = %s,
        xpm = %s,
        total_punts = %s,
        punt_yds = %s,
        pass_fds = %s,
        rsh_fds = %s,
        pen_fds = %s,
        total_fds = %s,
        thrd_down_conv = %s,
        thrd_down_att = %s,
        fourth_down_conv = %s,
        fourth_down_att = %s,
        penalties = %s,
        penalty_yds = %s,
        fmbl_lost = %s,
        int = %s,
        turnovers = %s,
        time_of_poss = %s
    WHERE team_id = %s AND week = %s AND year = %s
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, game_logs)
            connection.commit()
            logging.info(f"Successfully updated {len(game_logs)} team game logs in the database")

    except Exception as e:
        logging.error(
            f"An exception occurred while updating the team game logs: {game_logs}",
            exc_info=True,
        )
        raise e



def insert_team_rankings(rankings: list):
    """
    Insert record into 'team_ranks' with newly determined rankings 

    Args:
        rankings (list): list of rankings to persist 
    """

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
    

def insert_teams_odds(betting_odds: list, upcoming=False):
    """
    Functionality to insert historical betting odds into our DB 

    Args:
        betting_odds (list): list of historical betting odds 
    """

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


def update_teams_odds(betting_odds: list):
    """
    Update team betting odds within our database 

    Args:
        betting_odds (list): list of betting odds to update in DB 
    """
    
    sql = """
        UPDATE team_betting_odds 
        SET 
            game_over_under = %s,
            favorite_team_id = %s, 
            spread = %s
        WHERE 
            week = %s AND season = %s AND home_team_id = %s AND away_team_id = %s
    """

    params = [
        (
            odds["game_over_under"],
            odds["favorite_team_id"],
            odds["spread"],
            odds["week"],
            odds["year"],
            odds["home_team_id"],
            odds["away_team_id"],
        )
        for odds in betting_odds
    ]

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully updated {len(betting_odds)} team betting odds records in our database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while updating the following team betting odds: {betting_odds}",
            exc_info=True,
        )
        raise e
    


def update_team_game_log_with_results(update_records: list):
    """
    Update 'team_game_log' records with results of NFL games

    Args:
        update_records (list): list of records to update persisted records with
    """
    
    sql = """
        UPDATE team_game_log
        SET 
            result = %s,
            points_for = %s,
            points_allowed = %s,
            tot_yds = %s,
            pass_yds = %s,
            rush_yds = %s,
            opp_tot_yds = %s,
            opp_pass_yds = %s,
            opp_rush_yds = %s,
            pass_tds = %s,
            pass_cmp = %s,
            pass_att = %s,
            pass_cmp_pct = %s,
            rush_att = %s,
            rush_tds = %s,
            yds_gained_per_pass_att = %s,
            adj_yds_gained_per_pass_att = %s,
            pass_rate = %s,
            sacked = %s,
            sack_yds_lost = %s,
            rush_yds_per_att = %s,
            total_off_plays = %s,
            total_yds = %s,
            yds_per_play = %s,
            fga = %s,
            fgm = %s,
            xpa = %s,
            xpm = %s,
            total_punts = %s,
            punt_yds = %s,
            pass_fds = %s,
            rsh_fds = %s,
            pen_fds = %s,
            total_fds = %s,
            thrd_down_conv = %s,
            thrd_down_att = %s,
            fourth_down_conv = %s,
            fourth_down_att = %s,
            penalties = %s,
            penalty_yds = %s,
            fmbl_lost = %s,
            int = %s,
            turnovers = %s,
            time_of_poss = %s
        WHERE 
            team_id = %s AND year = %s AND season = %s
    """

    try:
        connection = get_connection()

        params = [
            (
                record["result"],
                record["points_for"],
                record["points_allowed"],
                record["tot_yds"],
                record["pass_yds"],
                record["rush_yds"],
                record["opp_tot_yds"],
                record["opp_pass_yds"],
                record["opp_rush_yds"],
                record["pass_tds"],
                record["pass_cmp"],
                record["pass_att"],
                record["pass_cmp_pct"],
                record["rush_att"],
                record["rush_tds"],
                record["yds_gained_per_pass_att"],
                record["adj_yds_gained_per_pass_att"],
                record["pass_rate"],
                record["sacked"],
                record["sack_yds_lost"],
                record["rush_yds_per_att"],
                record["total_off_plays"],
                record["total_yds"],
                record["yds_per_play"],
                record["fga"],
                record["fgm"],
                record["xpa"],
                record["xpm"],
                record["total_punts"],
                record["punt_yds"],
                record["pass_fds"],
                record["rsh_fds"],
                record["pen_fds"],
                record["total_fds"],
                record["thrd_down_conv"],
                record["thrd_down_att"],
                record["fourth_down_conv"],
                record["fourth_down_att"],
                record["penalties"],
                record["penalty_yds"],
                record["fmbl_lost"],
                record["int"],
                record["turnovers"],
                record["time_of_poss"],
                record["team_id"],
                record["year"],
                record["season"]
            )
            for record in update_records
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(f"Successfully updated {len(update_records)} team game log records in the database")
    except Exception as e:
        logging.error("An exception occurred while updating team_game_log", exc_info=True)
        raise e


def update_team_betting_odds_records_with_outcomes(update_records: list):
    """
    Update exisiting team_betting_odds records with the outcomes of each game
    
    Args:
        update_records (list): list of records to update 
    """

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



def insert_bye_week_rankings(team_bye_weeks: list, season: int): 
    """
    Insertion of bye week rankings for teams 

    Args:
        team_bye_weeks (list): records indicating bye weeks for teams 
        sesaon (int): the relevant season 
    """

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
    record = fetch_teams_home_away_wins_and_losses(year, team_id)
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
            cur.execute(punting_sql, punting_params)
            logging.info(f"Successfully inserted team_seasonal_punting_metrics record for team {team_id} season {season}")

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

    start_avg = team_stats.get('opp_start_avg', '')
    if start_avg and 'Own ' in start_avg:
        record['start_avg'] = float(start_avg.replace('Own ', ''))
    else:
        record['start_avg'] = 0

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


def insert_game_conditions(records: list):
    """
    Insert game condition records into the game_conditions table.

    Args:
        records (list): List of game condition records (dicts)
    """

    sql = """
        INSERT INTO game_conditions (
            season,
            week,
            home_team_id,
            visit_team_id,
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
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """

    try:
        connection = get_connection()

        params = [
            (
                int(record.get('season')),
                int(record.get('week')),
                int(record.get('home_team_id')),
                int(record.get('visit_team_id')),
                record.get('game_date'),
                int(record.get('game_time')) if record.get('game_time') else None,
                record.get('kickoff'),
                record.get('month'),
                record.get('start'),
                record.get('surface'),
                record.get('weather_icon'),
                float(record.get('temperature')) if record.get('temperature') else None,
                record.get('precip_probability'),
                record.get('precip_type'),
                float(record.get('wind_speed')) if record.get('wind_speed') else None,
                int(record.get('wind_bearing')) if record.get('wind_bearing') else None
            )
            for record in records
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info("Successfully inserted game condition records.")

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting game condition records: {records}",
            exc_info=True
        )
        raise e


def update_game_conditions(records: list):
    """
    Update existing game condition records in the game_conditions table.

    Args:
        records (list): List of game condition records (dicts)
    """
    
    sql = """
        UPDATE game_conditions
        SET 
            game_date = %s,
            game_time = %s,
            kickoff = %s,
            month = %s,
            start = %s,
            surface = %s,
            weather_icon = %s,
            temperature = %s,
            precip_probability = %s,
            precip_type = %s,
            wind_speed = %s,
            wind_bearing = %s
        WHERE
            season = %s AND
            week = %s AND
            home_team_id = %s AND
            visit_team_id = %s
    """

    try:
        connection = get_connection()

        params = [
            (
                record.get("game_date"),
                int(record.get("game_time")), 
                record.get("kickoff"),
                record.get("month"),
                record.get("start"),
                record.get("surface"),
                record.get("weather_icon"),
                float(record.get("temperature")),
                record.get("precip_probability"),
                record.get("precip_type"),
                float(record.get("wind_speed")),
                int(record.get("wind_bearing")),
                int(record.get("season")),
                int(record.get("week")),
                int(record.get("home_team_id")),
                int(record.get("visit_team_id"))
            )
            for record in records
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info("Successfully updated game condition records.")

    except Exception as e:
        logging.error(
            f"An exception occurred while updating game condition records: {records}",
            exc_info=True
        )
        raise e