from db.connection import get_connection
import logging
import pandas as pd
from service import player_service
from db.read.players import retrieve_player_demographics_record_by_pk



def insert_players(players: list):
    """
    Functionality to persist a particular player 

    Args: 
        players (list): list of players to persist 
    """

    try:
        for player in players:
            insert_player(player)

    except Exception as e:
        logging.error(
            f"An exception occurred while attempting to insert players {players}",
            exc_info=True,
        )
        raise e


def insert_player(player: dict):
    """
    Functionality to persist a single player 

    Args: 
        player (dict): player to insert into our db 
    """

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
            
            connection.commit()
            logging.info(
                f"Successfully updated {len(params)} player records with hashed names"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while updating player records with hashed names", exc_info=True
        )
        raise e



def update_player_pfr_availablity_status(player_ids: list, is_available: bool = False): 
    """
    Update player records to indicate they are not available in PFR (unable to find HREF)

    Args:
        player_ids (list): list of player_ids to update 
    """

    available = 1 if is_available else 0
        
    query = f"""
        UPDATE player 
        SET pfr_available = {available}
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
    

def insert_rb_player_game_logs(game_logs: list):
    """
    Functionality to persist multiple game logs for a RB 

    Args:
        game_logs (list): list of tuples to insert into player_game_logs 
    """

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


def insert_qb_player_game_logs(game_logs: list):
    """
    Functionality to persist multiple game logs for a QB

    Args:
        game_logs (list): list of tuples to insert into player_game_logs 
    """

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


def insert_wr_or_te_player_game_logs(game_logs: list):

    """
    Functionality to persist multiple game logs for a WR or TE

    Args:
        game_logs (list): list of tuples to insert into player_game_logs 
    """

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


def add_fantasy_points(fantasy_points: list):
    """
    Functionality to update player game log with calculated fantasy points

    Args:
        fantasy_points (list): list of items containing player_game_log PK and their fantasy points
    """

    sql = "UPDATE player_game_log SET fantasy_points = %s WHERE player_id = %s AND week = %s AND year = %s"

    try:
        connection = get_connection()

        params = [
            (log["fantasy_points"], log["player_id"], log["week"], log["year"])
            for log in fantasy_points
        ]  

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


def insert_upcoming_player_game_logs(game_logs: list):
    """
    Functionality to insert upcoming 'player_game_logs' into our database

    Args:
        game_logs (list): list of game logs to insert into DB 
    """ 

    sql = """
        INSERT INTO player_game_log (player_id, week, day, year, home_team, opp) 
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    params = [
        (
            game_log['player_id'],
            game_log['week'],
            game_log['day'],
            game_log['year'],
            game_log['home_team'],
            game_log['opp']
        )
        for game_log in game_logs
    ]
    
    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(params)} player_game_log records into our database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_game_log records: {game_logs}",
            exc_info=True,
        )
        raise e


def insert_player_props(player_props: dict, season: int):
    """
    Functionality to insert historical player props into our DB 

    Args:
        player_props (dict):  relevant season long player props to persist
        season (int): relevant season

    Returns:
        None
    """

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
    

def insert_upcoming_player_props(records: list):
    """
    Insert upcoming player_betting_odds records into our database 

    Args:
        records (list): list of records to insert 
    """

    sql = """
        INSERT INTO player_betting_odds (player_id, player_name, label, cost, line, week, season) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    params = [
        (
            record['player_id'],
            record['player_name'],
            record['label'],
            record['cost'],
            record['line'],
            record['week'],
            record['season']
        )
        for record in records
    ]
    
    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted {len(params)} upcoming player_betting_odds into our database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following upcoming player betting odds: {records}",
            exc_info=True,
        )
        raise e 


def update_upcoming_player_props(records: list):
    """
        Update upcoming player_betting_odds records in our database 

        Args:
            records (list): list of records to update
    """
    
    sql = """
        UPDATE player_betting_odds 
        SET 
            cost = %s,
            line = %s
        WHERE 
            week = %s AND season = %s AND label = %s AND player_id = %s
    """
    
    params = [
        (
            record['cost'],
            record['line'],
            record['week'],
            record['season'],
            record['label'],
            record['player_id']
        )
        for record in records
    ]
    
    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully updated {len(params)} upcoming player_betting_odds in our database"
            )

    except Exception as e:
        logging.error(
            f"An exception occurred while updating the following upcoming player betting odds: {records}",
            exc_info=True,
        )
        raise e 

    
def update_player_game_log_with_results(update_records: list):
    """
    Update 'player_game_log' records with their respective results 

    Args:
        update_records (list): list of records to updated persisted records in db with
    """

    sql = """
        UPDATE player_game_log
        SET 
            result = %s,
            points_for = %s,
            points_allowed = %s,
            completions = %s,
            attempts = %s,
            pass_yd = %s,
            pass_td = %s,
            interceptions = %s,
            rating = %s,
            sacked = %s,
            rush_att = %s,
            rush_yds = %s,
            rush_tds = %s,
            tgt = %s,
            rec = %s,
            rec_yd = %s,
            rec_td = %s,
            snap_pct = %s,
            fantasy_points = %s,
            off_snps = %s
        WHERE 
            year = %s AND player_id = %s AND week = %s
    """

    try:
        connection = get_connection()

        params = [
            (
                record["result"],
                record["points_for"],
                record["points_allowed"],
                record["completions"],
                record["attempts"],
                record["pass_yd"],
                record["pass_td"],
                record["interceptions"],
                record["rating"],
                record["sacked"],
                record["rush_att"],
                record["rush_yds"],
                record["rush_tds"],
                record["tgt"],
                record["rec"],
                record["rec_yd"],
                record["rec_td"],
                record["snap_pct"],
                record["fantasy_points"],
                record["off_snps"],
                record["year"],
                record["player_id"],
                record["week"]
            )
            for record in update_records
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(f"Successfully updated {len(update_records)} player game log records in the database")
    except Exception as e:
        logging.error("An exception occurred while updating player_game_log", exc_info=True)
        raise e


def update_player_teams_records_end_dates(player_teams_records: list): 
    """
    Functionality to end date 'player_teams' records in our database 

    Args:
        player_teams_records (list): records requiring updates
    
    """
    sql = f"""
            UPDATE player_teams 
            SET end_wk = %s
            WHERE player_id = %s AND team_id = %s AND season = %s
         """

    try:
        connection = get_connection()

        params = [
            (
                record["end_wk"],
                record["player_id"],
                record["team_id"],
                record["season"]
            )
            for record in player_teams_records 
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully updated {len(player_teams_records)} player_teams records in the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while updating the following player_teams records in our database: {player_teams_records}",
            exc_info=True,
        )
        raise e

def insert_player_teams(player_teams_records: list): 
    """
    Insert new player teams records into datbaase 

    Args:
        player_teams_records (list): records to persist 
    """

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


def update_player_depth_chart_postion(player_depth_chart: list): 
    """
    Functionality to update player depth chart record positions in our DB 

    ArgsL
        player_depth_chart (list): players depth charts to update
    """

    sql = """
            UPDATE player_depth_chart 
            SET depth_chart_pos = %s 
            WHERE player_id = %s AND season = %s AND week = %s
         """

    try:
        connection = get_connection()

        params = [
            (
                record["depth_chart_pos"],
                record["player_id"],
                record["season"],
                record["week"]
            )
            for record in player_depth_chart
        ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully updated {len(player_depth_chart)} player_depth_chart records in the database"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while updating the following player_depth_chart records in our db: {player_depth_chart}",
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



def insert_player_demographics(df: pd.DataFrame):
    """
    Insert player demographics (i.e age, height, weight) for a given season

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
    ids = set()  # track player IDs

    try:
        connection = get_connection()

        for _, row in df.iterrows(): 
            try:
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
            if retrieve_player_demographics_record_by_pk(param[1], param[0]) is None:
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


def insert_player_injuries(records: list):

    """
    Insert new records into our player_injuries db table 

    Args:
        records (list): list of records to insert 
    """

    sql = """
    INSERT INTO player_injuries(
        player_id, week, season, injury_loc, wed_prac_sts, thurs_prac_sts, fri_prac_sts, off_sts
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        params = [
        (
            int(record.get("player_id")), int(record.get('week')), int(record.get('season')),
            record.get('injury_loc'),
            record.get('wed_prac_sts'),
            record.get('thurs_prac_sts'),
            record.get('fri_prac_sts'),
            record.get('off_sts'),
        )
        for record in records ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully inserted player_injuries records"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while inserting the following player_injuries records into our db: {records}",
            exc_info=True,
        )
        raise e


def update_player_injuries(records: list):

    """
    Update existing records in our player_injuries db table 

    Args:
        records (list): list of records to insert 
    """

    sql = """
    UPDATE player_injuries
    SET 
        injury_loc = %s,
        wed_prac_sts = %s,
        thurs_prac_sts = %s, 
        fri_prac_sts = %s,
        off_sts = %s
    WHERE 
        player_id = %s AND week = %s AND season = %s
    """

    try:
        connection = get_connection()

        params = [
        (
            record.get('injury_loc'),
            record.get('wed_prac_sts'),
            record.get('thurs_prac_sts'),
            record.get('fri_prac_sts'),
            record.get('off_sts'),
            int(record.get("player_id")), int(record.get('week')), int(record.get('season'))
        )
        for record in records ]

        with connection.cursor() as cur:
            cur.executemany(sql, params)
            connection.commit()
            logging.info(
                f"Successfully updated player_injuries records"
            )
    except Exception as e:
        logging.error(
            f"An exception occurred while updating the following player_injuries records into our db: {records}",
            exc_info=True,
        )
        raise e
    



def insert_player_dob(player_id: int, dob: str):
    """
    Insert players date of birth 

    Args:
        player_id (int): the player id corresponding to player being inserted 
        dob (str): the players date of birth 
    """

    sql = """
        UPDATE player
        SET dob = %s 
        WHERE player_id = %s
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (dob, player_id))
            connection.commit()
            logging.info(f"Successfully updated player with ID {player_id} with the following date of birth: {dob}")

    except Exception as e:
        logging.error(
            f"An exception occurred while updating 'player' record with ID {player_id} with the following DOB {dob}",
            exc_info=True
        )
        raise e  


def insert_player_demographics(player_id: int, season: int, age: int, height: float, weight: float):
    """
    Insert a new record into the player_demographics table.

    Args:
        player_id (int): The unique ID of the player.
        season (int): The season year for which the demographic data is recorded.
        age (int): The age of the player during the specified season.
        height (float): The height of the player (in inches or cm based on your schema).
        weight (float): The weight of the player (in pounds or kg based on your schema).
    """

    sql = """
        INSERT INTO player_demographics (player_id, season, age, height, weight)
        VALUES (%s, %s, %s, %s, %s)
    """

    try:
        connection = get_connection()

        with connection.cursor() as cur:
            cur.execute(sql, (player_id, season, age, height, weight))
            connection.commit()
            logging.info(f"Successfully inserted player_demographics record for player_id={player_id}, season={season}")

    except Exception as e:
        logging.error(
            f"An error occurred while inserting into player_demographics for player_id={player_id}, season={season}",
            exc_info=True
        )
        raise e