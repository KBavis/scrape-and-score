from config import configure_logging
from scraping import scrape_fantasy_pros
from scraping import pfr_scraper as pfr
from db import (
    connection,
    insert_teams,
    fetch_all_teams,
    insert_players,
    fetch_all_players,
)
from config import load_configs, get_config
from datetime import datetime
from service import player_game_logs_service, team_game_logs_service
from data import preprocess
from models.lin_reg import LinReg
from util import args
import logging
from predictions import prediction
from scraping import rotowire_scraper


"""
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
"""


def main():

    cl_args = args.parse()

    try:
        # init configs, logging, and db connection
        configure_logging()
        load_configs()
        connection.init()
        start_time = datetime.now()
        year = get_config("nfl.current-year")
        team_names_and_ids = []

        # first time application invoked
        if cl_args.new:
            logging.info(
                "First application run; persisting all configured NFL teams to our database"
            )

            template_url = get_config("website.fantasy-pros.urls.depth-chart")
            teams = [
                team["name"] for team in get_config("nfl.teams")
            ]  # extract teams from configs
            team_names_and_ids = insert_teams(teams)

            players = scrape_fantasy_pros(template_url, team_names_and_ids)
            logging.info(
                f"Successfully fetched {len(depth_charts)} unique fantasy relevant players and their corresponding teams"
            )

            # insert players into db
            logging.info("Inserting fantasy relevant players into db")
            insert_players(players)

            depth_charts = fetch_all_players()

            team_metrics, player_metrics = pfr.scrape_all(depth_charts, teams)
            logging.info(
                f"Successfully retrieved metrics for {len(team_metrics)} teams and {len(player_metrics)} players"
            )

            # insert into player_game_log & team_game_log
            player_game_logs_service.insert_multiple_players_game_logs(
                player_metrics, depth_charts
            )
            team_game_logs_service.insert_multiple_teams_game_logs(
                team_metrics, team_names_and_ids
            )

            # calculate fantasy points for each week
            player_game_logs_service.calculate_fantasy_points(False, year)

            # calculate all teams rankings
            team_game_logs_service.calculate_all_teams_rankings(year)
        # application invoked for recent week
        elif cl_args.recent:
            logging.info(
                "Teams previously persisted to database; skipping insertion and fetching teams from database"
            )
            team_names_and_ids = fetch_all_teams()

            # TODO: FFM-64 - Check for depth chart changes to determine if need to make updates to persisted data
            logging.info("All players persisted; skipping insertion")

            depth_charts = fetch_all_players()

            logging.info(
                "All previous games fetched; fetching metrics for most recent week"
            )
            team_metrics, player_metrics = pfr.scrape_recent()
            logging.info(
                f"Successfully retrieved most recent game log metrics for {len(team_metrics)} teams and {len(player_metrics)} players"
            )

            # insert into player_game_log & team_game_log
            player_game_logs_service.insert_multiple_players_game_logs(
                player_metrics, depth_charts
            )
            team_game_logs_service.insert_multiple_teams_game_logs(
                team_metrics, team_names_and_ids
            )
            logging.info("Successfully persisted most recent player & game log metrics")

            # calculate fantasy points for recent week
            player_game_logs_service.calculate_fantasy_points(True, year)

            # calculate all teams rankings based on new data
            team_game_logs_service.calculate_all_teams_rankings(year)
        else:
            logging.info("--recent nor --new flag passed in; skipping scraping...")

        # pre-process persisted data
        (
            qb_pre_processed_data,
            rb_pre_processed_data,
            wr_pre_processed_data,
            te_pre_processed_data,
        ) = preprocess.pre_process_data()

        # generate our position specific regressions
        linear_regressions = LinReg(
            qb_pre_processed_data,
            rb_pre_processed_data,
            wr_pre_processed_data,
            te_pre_processed_data,
        )
        linear_regressions.create_regressions()

        if cl_args.upcoming:
            rotowire_scraper.scrape_upcoming()
        elif cl_args.update_odds:
            rotowire_scraper.update_recent_betting_records()
        elif cl_args.historical:
            rotowire_scraper.scrape_all()

        # determine if we want to test our regression
        if cl_args.test:
            linear_regressions.test_regressions()

        # determine inputs
        if cl_args.single_player:
            # prompt user to input player name & matchup
            logging.info("Prompting user to input player name & matchup ...")
            prediction.make_single_player_prediction(linear_regressions)

        elif cl_args.all_players:
            # fetch upcoming matchups and make predictions
            logging.info(
                "Fetching upcoming matchups for fantasy relevant players and predicting their fantasy points ..."
            )

        logging.info(
            f"Application took {(datetime.now() - start_time).seconds} seconds to complete"
        )

    except Exception as e:
        logging.error(f"An exception occured while executing the main script: {e}")
    finally:
        connection.close_connection()


# entry point
if __name__ == "__main__":
    main()
