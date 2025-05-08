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
from scraping import rotowire_scraper, our_lads, betting_pros
from data.dataset import FantasyDataset
from torch.utils.data import DataLoader
from models.neural_net import NeuralNetwork
from models import optimization, post_training
from constants import TRAINING_CONFIGS
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

            positions = ['RB', 'QB', 'TE', 'WR']
            required_models = [
                'rb_model.pth',
                'qb_model.pth',
                'te_model.pth',
                'wr_model.pth'
            ] 

            # check if models exists and  that we do not want to retrain models
            directory = "models/nn/{}"
            if all(os.path.exists(directory.format(model)) for model in required_models) and cl_args.train == False:
                rb_nn = torch.load('rb_model.pth', weights_only=False)
                qb_nn = torch.load('qb_model.pth', weights_only=False)
                wr_nn = torch.load('wr_model.pth', weights_only=False)
                te_nn = torch.load('te_model.pth', weights_only=False)
            else:

                # pre-process training & testing data
                df = nn_preprocess.preprocess()
                position_features = [f'position_{position}' for position in positions]
                
                # data set up & training loop for each position
                for position in positions:
                    logging.info(f'Extracting {position} pre-processed data into training/testing data sets')

                    # extract records relevant to particular position 
                    position_feature = f'position_{position}'
                    position_specific_df = df[df[position_feature] == 1].copy()

                    # drop cateogorical features for determining positions
                    position_specific_df.drop(columns=position_features, inplace=True)

                    # split into training & testing data frames 
                    num_records = len(position_specific_df)
                    training_length = int(num_records* 0.8)
                    training_df = position_specific_df.iloc[: training_length]
                    testing_df = position_specific_df.iloc[training_length : num_records]

                    # perform feature selection on model 
                    selected_features = nn_preprocess.feature_selection(position_specific_df, position)

                    # cache selected features 
                    directory = "data/inputs"
                    timestamp = start_time.strftime('%Y%m%d_%H%M%S')

                    os.makedirs(directory, exist_ok=True)
                    with open(f'{directory}/{position}_inputs_{timestamp}.txt', 'w') as f: 
                        for col in selected_features: 
                            f.write(col + '\n')
                    
                    # create datasets & data loaders 
                    training_data_set = FantasyDataset(training_df[selected_features + ["fantasy_points"]])
                    testing_data_set = FantasyDataset(testing_df[selected_features + ["fantasy_points"]])
                    test_data_loader = DataLoader(testing_data_set, batch_size=TRAINING_CONFIGS[position]['Batch Size'], shuffle=False) 
                    train_data_loader = DataLoader(training_data_set, batch_size=TRAINING_CONFIGS[position]['Batch Size'], shuffle=True) 

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    print(f"Using the following device to train: {device}")            

                    nn = NeuralNetwork(input_dim = len(selected_features), position=position).to(device)
                    print(f"Attempting to train {position} Specific Neural Network:\n\nLength of Training Data: {len(training_data_set)}\n\nNumber of Inputs: {len(selected_features)}\n\nModel: {nn}\n\nList of Inputs: {selected_features}")

                    # start optimization loop
                    learning_rate = TRAINING_CONFIGS[position]['Learning Rate']
                    optimization.optimization_loop(train_data_loader, test_data_loader, nn, device, learning_rate)

                    directory = "models/nn"
                    os.makedirs(directory, exist_ok=True)
                    torch.save(nn, f'{directory}/{position.lower()}_model.pth')

                    # determine feature importance 
                    post_training.feature_importance(nn, training_data_set, position, device)
            


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
