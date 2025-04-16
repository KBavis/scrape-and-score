from config import configure_logging
from scraping import scrape_fantasy_pros
from scraping import pfr_scraper as pfr
from scraping import football_db 
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
from data import linreg_preprocess, nn_preprocess
from models.lin_reg import LinReg
from util import args
import logging
from predictions import prediction
from scraping import rotowire_scraper, our_lads, betting_pros
from data.dataset import FantasyDataset
from torch.utils.data import DataLoader
from models.neural_net import NeuralNetwork
from models import optimization, post_training
import torch
import os


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
        #TODO: Remove new args and any deprecated functionality after removing (i.e scrape_all, scrape fantasy pros, etc)
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
            #TODO: if this logic is not removed in future, it needs to be updated to have player be key in the dict rather than player_name
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
        

        # collect multiple years worth of data 
        #TODO: Refactor above approach and this approach to be unified
        if cl_args.collect_data: 
            
            start_year, end_year = cl_args.collect_data
            logging.info(f"Attempting to collect relevant player and team data from the year {start_year} to {end_year}")

            # account for teams potentially not being persisted yet
            teams = fetch_all_teams() 
            if not teams:
                teams = [
                    team["name"] for team in get_config("nfl.teams")
                ]             
                insert_teams(teams)
            
            #TODO: Ensure that this logic is resilient for issues where records already exist

            # fetch & persist player records and their corresponding player_teams recrods
            our_lads.scrape_and_persist(start_year, end_year)

            # fetch & persist player and team game logs 
            pfr.scrape_historical(start_year, end_year) 

            # calculate & persist fantasy points, TODO: Account for fumbles and 2 PT conversions for better accuracy
            player_game_logs_service.calculate_fantasy_points(False, start_year, end_year) 

            # calculate & persist team rankings for relevant years
            for curr_year in range(start_year, end_year):
                team_game_logs_service.calculate_all_teams_rankings(curr_year)


            # fetch & persist team betting odds & game conditions for relevant seasons
            rotowire_scraper.scrape_all(start_year, end_year) 

            # fetch & persist player betting odds for relevant season
            for curr_year in range(start_year, end_year + 1):
                betting_pros.fetch_historical_odds(curr_year)

            pfr.fetch_teams_and_players_seasonal_metrics(start_year, end_year)

            pfr.scrape_player_advanced_metrics(start_year, end_year)

            football_db.scrape_historical(start_year, end_year)

        # scrape relevant betting odds based on specified arg
        if cl_args.upcoming:
            rotowire_scraper.scrape_upcoming()
        elif cl_args.update_odds:
            rotowire_scraper.update_recent_betting_records()
        elif cl_args.historical:
            rotowire_scraper.scrape_all()

        
        if cl_args.lin_reg:
            # pre-process persisted data
            (
                qb_pre_processed_data,
                rb_pre_processed_data,
                wr_pre_processed_data,
                te_pre_processed_data,
            ) = linreg_preprocess.pre_process_data()
            
            logging.info('Successfully pre-processed all data')

            # generate our position specific regressions
            linear_regressions = LinReg(
                qb_pre_processed_data,
                rb_pre_processed_data,
                wr_pre_processed_data,
                te_pre_processed_data,
            )
            linear_regressions.create_regressions()

            # test regressions 
            linear_regressions.test_regressions()
        elif cl_args.nn:
            
            # Check if Neural Network model already exists and that we don't want to re-tain model
            if os.path.exists('model.pth') and cl_args.train == False:
                nn = torch.load('model.pth', weights_only=False)
            else:
                df = nn_preprocess.preprocess()
                
                # split into training & testing dataframes 
                num_rows = len(df)
                training_length = int(num_rows * 0.8)
                
                training_df = df.iloc[: training_length]
                
                testing_df = df.iloc[training_length : num_rows]
                
                training_data_set = FantasyDataset(training_df)
                testing_data_set = FantasyDataset(testing_df)
                
                test_data_loader = DataLoader(testing_data_set, batch_size=256, shuffle=False) # TODO: determine appropiate batchsize 
                train_data_loader = DataLoader(training_data_set, batch_size=256, shuffle=True) # TODO: determine appropiate batchsize 
                
                # print(f"Accelator Available: {torch.accelerator.is_available()}")
                
                number_inputs = df.shape[1] - 1
                columns = list(df.columns)
                inputs = [col for col in columns if col != "fantasy_points"]
                nn = NeuralNetwork(input_dim = number_inputs)  
                print(f"Attempting to train Neural Network:\n\nNumber of Inputs: {number_inputs}\n\nModel: {nn}\n\nList of Inputs: {inputs}")

                # start optimization loop
                optimization.optimization_loop(train_data_loader, test_data_loader, nn)

                torch.save(nn, 'model.pth')

                # determine feautre importance 
                post_training.feature_importance(nn, training_data_set)

                
            


        # determine type of predicitions to make
        #TODO: Fix single player predicition for Linear Regressions & add logic for predicitng all upcoming players points 
        #TODO: Add single player prediciton logic for Neural Network & add logic for predicitng all upcoming players points
        if cl_args.single_player:
            # prompt user to input player name & matchup
            logging.info("Prompting user to input player name & matchup ...")
            # prediction.make_single_player_prediction(linear_regressions) TODO: Uncomment me 

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
