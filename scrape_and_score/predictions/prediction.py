from . import input
import pandas as pd
from db import fetch_data
from data import preprocess
import logging
from sklearn.preprocessing import StandardScaler
from models.lin_reg import LinReg
import numpy as np
from config import props

'''
Functionality to make a prediction for a single players # of fantasy points based on their matchup

Args:
   linear_regression (LinReg): linear regression model

Returns:
   None
'''
def make_single_player_prediction(linear_regressions: LinReg):
   week, player_name = input.get_user_input()
   season = props.get_config('nfl.current-year')
   
   # fetch model corresponding to players position 
   logging.info(f'Fetching independent variables corresponding to player {player_name}...')
   independent_vars = fetch_data.fetch_inputs_for_prediction(week,season,player_name)
   
   logging.info(f'Calculating ratio rank for player {player_name}')
   inputs_with_ratio_rank = calc_rankings_ratio(independent_vars)
   
   logging.info(f'Applying logarithm transformation to inputs for player {player_name}')
   transformed_inputs, _, _, _= preprocess.transform_data(inputs_with_ratio_rank)
   
   position = transformed_inputs['position'].iloc[0]
   
   
   col_mappings = {
      "QB": ['log_avg_fantasy_points', 'log_ratio_rank', 'game_over_under', 'is_favorited'],
      "RB": ['log_avg_fantasy_points', 'log_ratio_rank', 'game_over_under', 'is_favorited'],
      "WR": ['log_avg_fantasy_points', 'log_ratio_rank', 'game_over_under'],
      "TE": ['log_avg_fantasy_points', 'game_over_under']
   }
   inputs = transformed_inputs[col_mappings[position]]
   
   scaled_inputs = scale_inputs(inputs)
   
   predict(linear_regressions,scaled_inputs, position)
   
   
'''
Make prediciton based on players posiiton and scaled inputs
''' 
def predict(linear_regressions: LinReg, scaled_inputs: np.array, position: str):
   # get corresponding LinearRegression model 
   lin_reg_model = None
   if position == 'RB':
      lin_reg_model = linear_regressions.rb_regression
   elif position == 'QB':
      lin_reg_model = linear_regressions.qb_regression
   elif position == 'TE':
      lin_reg_model = linear_regressions.te_regression
   else:
      lin_reg_model = linear_regressions.qb_regression
   
   value = lin_reg_model.predict(scaled_inputs)
   logging.info(f'Predicted # of fantasy points: {np.exp(value)}')
   
'''
Functionality to scale inputs 
   
Args:
   df (pd.DataFrmae): data frame to scale inputs for 
   
Returns
   inputs (pd.DataFrame): inputs from data frame
''' 
def scale_inputs(df: pd.DataFrame): 
   scaler = StandardScaler() 
   scaler.fit(df)
   scaled = scaler.transform(df)
   return scaled

'''
Calculate players ranking ratio based on offensive rank & opposing defense ranking

Args:
   inputs (pd.DataFrame): data frame to calc rankings for 

Returns:
   inputs (pd.DataFrame): data frame with calculated ratio rank
'''
def calc_rankings_ratio(inputs: pd.DataFrame):
   if inputs['position'].iloc[0] == 'RB':
      rush_ratio_rank = inputs['def_rush_rank'] / inputs['off_rush_rank']
      inputs['rush_ratio_rank'] = rush_ratio_rank
   else :
      pass_ratio_rank = inputs['def_pass_rank'] / inputs['off_pass_rank']
      inputs['pass_ratio_rank'] = pass_ratio_rank
   
   return inputs
