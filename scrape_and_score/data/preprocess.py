from db import fetch_data
import pandas as pd


'''
Main functionality of module to kick of data fetching and pre-procesing 

Args:
   None 

Returns:
   None
'''
def pre_process_data(): 
   df = get_data()
   print(df)

'''
Filter out records where fantasy points are zero 

Args:
   df (pd.DataFrame): dataframe to filter out records 

Returns:
   filtered_df (pd.DataFrame): data frame with records filtered
'''
# def filter_irrelevant_records(df: pd.DataFrame):
   

'''
Addition of avg_fantasy_points feature so historical data is included in model 

Args:
   df (pd.DataFrame): dataframe to add avg_fantasy_points to 

Returns:
   df (pd.DataFrame): updated data frame 
'''
def include_averages(df: pd.DataFrame):
   player_avg_points = df.groupby('player_id')['fantasy_points'].mean().round(2) 
   df['avg_fantasy_points'] = df['player_id'].map(player_avg_points)
   return df

'''
Functionality to retrieve pandas df, containing independent & dependent variable(s)

Args:
   None

Returns:
   df (pd.DataFrame): data frame containing inputs/outputs for linear regression model
'''
def get_data():
   df = fetch_data.fetch_independent_and_dependent_variables_for_mult_lin_regression()
   updated_df = include_averages(df)
   return updated_df
   