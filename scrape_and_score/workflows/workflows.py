from nnutils import optimization
from scraping import pfr, rotowire
from scraping import our_lads, football_db, betting_pros, espn
from service import team_game_logs_service, player_game_logs_service
from models.lin_reg import LinReg
from data import nn_preprocess, linreg_preprocess
import torch
import os
import logging
from data.dataset import FantasyDataset
from torch.utils.data import DataLoader
from datetime import datetime
from constants import TRAINING_CONFIGS
from models.neural_net import NeuralNetwork
from nnutils import post_training
from . import utils 
from nnutils import prediction
from config import props


FINAL_WEEK = 18

rb_nn = None
qb_nn = None
wr_nn = None
te_nn = None



def predict(week: int, season: int, model: str):
    """
    Generate and output the top 40 players by fantasy points for each position 
    for the specified week and season using the given model.

    Args:
        week (int): relevant week to generate predictions for 
        season (int): relevant seaon to generate predictions for 
        model (str): model to utilize to generate predictions (either 'nn' or 'lin_reg')
    """

    if model != 'nn' and model != 'lin_reg':
        raise Exception("Only 'nn' or 'lin_reg' are valid aguments to invoke 'predict' workflow")
    


    #TODO: Create linear regression prediction logic 
    if model == 'lin_reg':
        raise Exception('Linear regression prediction functionality is currently not implemented')
    

    # fetch relevant prediction data 
    df = nn_preprocess.preprocess(week, season)

    positions = ['QB', 'RB', 'WR', 'TE']

    # create mappings 
    feature_mapping = utils.get_position_features()
    model_mapping = {
        'QB': qb_nn,
        'RB': rb_nn,
        'WR': wr_nn,
        'TE': te_nn
    }
    
    # iterate through relevant positions
    predictions = {}
    for position in positions: 

        # extract position relevant data 
        position_feature = f'position_{position}'
        position_specific_df = df[df[position_feature] == 1].copy()

        # extract players associated with each individaul prediction
        players = position_specific_df['player_id']

        # extract most recent relevant features corresponding to position
        position_features = feature_mapping[position]

        # ensure columns are in correct order and all features are available that were used to train nn
        position_specific_df = nn_preprocess.add_missing_features(position_specific_df, position_features)

        # seperate out inputs 
        prediction_data = position_specific_df[position_features]
        X = torch.from_numpy(nn_preprocess.scale_and_transform(prediction_data)).float()

        predictions[position] = prediction.generate_predictions(position, week, season, model_mapping[position], players, X)

    prediction.log_predictions(predictions, week, season)



def upcoming(week: int, season: int):
    """
    Invoke necessary functionality to scrape & persist player / team data for upcoming week

    Args:
        week (int): upcoming week to extract functionality for 
        season (int): the season this data corresponds to 
    """

    # scrape & persist player records (update records if necessary)
    is_update = utils.is_player_records_persisted(season)
    our_lads.scrape_and_persist_upcoming(season, week, is_update=is_update)
       
    
    # # scrape & persist player demographics (if necessary)
    if not utils.is_player_demographics_persisted(season):
        pfr.scrape_and_persist_player_demographics(season)

    # scrape & update upcoming NFL games
    espn.scrape_upcoming_games(season, week)

    # generate mapping of team_ids / player_ids / game_date 
    game_mapping = utils.generate_game_mapping(season, week)

    # filter games already played 
    relevant_games = utils.filter_completed_games(game_mapping)

    # extract relevant IDs
    all_team_ids = [team_id for game in relevant_games for team_id in game["team_ids"]]
    all_player_ids = [player_id for game in relevant_games for player_id in game['player_ids']]

    #TODO: Filter out 'pfr_unavailable' players 

    # insert strubbed player game logs if necesary 
    utils.add_stubbed_player_game_logs(all_player_ids, week, season)

    # scrape upcoming team betting odds & game conditions 
    rotowire.scrape_upcoming(week, season, all_team_ids)

    # scrape player injuries 
    football_db.scrape_upcoming(week, season, all_player_ids) #TODO: Validate this site is updated throughout week of the NFL game 

    # scrape player betting odds & game conditions 
    betting_pros.fetch_upcoming_player_odds_and_game_conditions(week, season, all_player_ids)


def historical(start_year: int, end_year: int):
    """
    Invoke necessary functionality to scrape & persist player / team data across multiple seasons

    TODO: Ensure this is resilient for situations where some data may alrady have been persisted


    Args:
        start_year (int): year to start scraping data from
        end_year (int): year to stop scraping data from
    """

    # scrape & persist player records and their corresponding player_teams recrods
    our_lads.scrape_and_persist(start_year, end_year)

    # scrape & persist player and team game logs 
    pfr.scrape_historical(start_year, end_year) 

    # calculate & persist fantasy points, TODO: Account for fumbles and 2 PT conversions for better accuracy
    player_game_logs_service.calculate_fantasy_points(False, start_year, end_year) 

    # calculate & persist team rankings for relevant years
    for curr_year in range(start_year, end_year):
        team_game_logs_service.calculate_all_teams_rankings(curr_year)


    # fetch & persist team betting odds & game conditions for relevant seasons
    rotowire.scrape_all(start_year, end_year) 

    # fetch & persist player betting odds for relevant season
    for curr_year in range(start_year, end_year + 1):
        betting_pros.fetch_historical_odds(curr_year)


    # scrape & persist teams/players seasonal metrics 
    pfr.fetch_teams_and_players_seasonal_metrics(start_year, end_year)

    # scrape & persit player weekly advanced metrics 
    pfr.scrape_player_advanced_metrics(start_year, end_year)

    # scrape & persist player injuries
    football_db.scrape_historical(start_year, end_year)



def results(week: int, season: int):
    """
    Invoke necessary functionality to scrape necessary updates to records based on game outcomes. 

    Args:
        week (int): upcoming week to extract functionality for 
        season (int): the season this data corresponds to 
    """

    # update 'team_game_log' and 'player_game_log' records with results
    pfr.update_game_logs_and_insert_advanced_metrics(week, season)
    
    # update player game log records with fantasy points 
    player_game_logs_service.calculate_fantasy_points(season, season, week=week) 

    # insert 'team_ranks' records based on outcomes 
    team_game_logs_service.calculate_all_teams_rankings(season, week)

    # update 'team_betting_odds' records with results
    rotowire.update_recent_betting_records(week, season)

    # insert seasonal rankings if week 18 and not inserted
    if week == props.get_config('final-week'):
        if not utils.are_team_seasonal_metrics_persisted(season) or not utils.are_player_seasonal_metrics_persisted(season):
            pfr.fetch_teams_and_players_seasonal_metrics(season, season)


def linear_regression():
    """
    Pre-process relevant data, generate linear regression, and test our linear regression against test data

    TODO: This functionality is incomplete as the focus was shifted towards using a NeuralNetwork
    """

    # pre-process persisted data
    (
        qb_pre_processed_data,
        rb_pre_processed_data,
        wr_pre_processed_data,
        te_pre_processed_data,
    ) = linreg_preprocess.pre_process_data()

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



def neural_network(should_train: bool, start_time: datetime):
    """
    Train/test Neural Network model for predciitng players fantasy points for a given week 

    Args:
        should_train (bool): flag to indicate if we should re-train our models
        start_time (datetime): start time of our application
    """
    
    positions = ['RB', 'QB', 'TE', 'WR']
    required_models = [
        'rb_model.pth',
        'qb_model.pth',
        'te_model.pth',
        'wr_model.pth'
    ] 
    global rb_nn, qb_nn, wr_nn, te_nn

    # check if models exists and  that we do not want to retrain models
    directory = "models/nn/{}"
    if all(os.path.exists(directory.format(model)) for model in required_models) and should_train == False:
        rb_nn = torch.load(f'{directory.format('rb_model.pth')}', weights_only=False)
        qb_nn = torch.load(f'{directory.format('qb_model.pth')}', weights_only=False)
        wr_nn = torch.load(f'{directory.format('wr_model.pth')}', weights_only=False)
        te_nn = torch.load(f'{directory.format('te_model.pth')}', weights_only=False)
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