from . import input
import pandas as pd
from db import fetch_data

'''
Functionality to make a prediction for a single players # of fantasy points based on their matchup

Args:
   None 

Returns:
   None
'''
def make_single_player_prediction():
   team_name, player_name = input.get_user_input()
   
   # fetch model corresponding to players position 
   inputs = fetch_data.fetch_inputs_for_prediction(team_name, player_name)
   
   # call predict 
   


def get_inputs_for_prediction(): 
   # fetch player db record 
   
   # fetch player game log db records 
   
   # fetch players corresponding team db record 
   
   # fetch opposing teams db record
   return None

'''
Apply logarithm transformations to inputs 

Args:
   df (pd.DataFrame): data frame with inputs to transform 

Returns:
   transformed_df (pd.DataFrame): transformed data frame
'''
def transform_inputs():
   return None

'''
Scale inputs in order to properly predict # of fantasy points 

Args:
   inputs (df): data frame to scale 

Returns:
   scaled_inputs (df): scaled inputs
'''
def scale_inputs(inputs:pd.DataFrame):
   return None