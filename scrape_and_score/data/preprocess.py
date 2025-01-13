from db import fetch_data
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os


'''
Main functionality of module to kick of data fetching and pre-procesing 

Args:
   None 

Returns:
   None
'''
def pre_process_data(): 
   sns.set_theme()
   
   df = get_data()
   
   filtered_df = filter_data(df)
   
   qb_data, rb_data, te_data, wr_data = split_data_by_position(filtered_df)
   
   transformed_qb_data, transformed_rb_data, transformed_wr_data, transformed_te_data = transform_data(qb_data, rb_data, wr_data, te_data)
   tranformed_data = [transformed_qb_data, transformed_rb_data, transformed_wr_data, transformed_te_data]
   
   ols_validated_data = [] 
   for df in tranformed_data:
      df.drop(columns=['player_id', 'fantasy_points', 'off_rush_rank', 'off_pass_rank', 'def_rush_rank', 'def_pass_rank']) # drop un-needed columns
      ols_validated_data.append(validate_ols_assumptions(df))
   
   cols = ['log_fantasy_points', 'log_avg_fantasy_points', 'log_ratio_rank']
   preprocessed_data = [df[cols] for df in ols_validated_data] 
   
   # return tuple of pre-processed data
   return preprocessed_data[0], preprocessed_data[1], preprocessed_data[2], preprocessed_data[3]



'''
Functionality to vlaidate multicollinearity & other OLS assumptions 

Args:
   df (pd.DataFrame): data frame to validate 

Returns:
   updated_df (pd.DataFrame): updated df (if there weren't anym then return original)
'''
def validate_ols_assumptions(df: pd.DataFrame):
   variables = df[['log_avg_fantasy_points', 'log_ratio_rank']]
   
   vif = pd.DataFrame() 
   
   vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
   vif["Features"] = variables.columns
   
   
   features_to_remove = vif[vif['VIF'] > 5]["Features"].tolist()
   if features_to_remove:
      logging.info(f'Removing the following columns due to high VIFs: [{features_to_remove}]')
   else:
      logging.info("No features need to be removed to validate OLS assumptions; returning original DataFrame")
      
   return df.drop(columns=features_to_remove)
   

'''
Split dataframes into position specific data due to fantasy points 
vary signficantly by position and invoke functionality to get ranking ratios

Args:
   df (pd.DataFrame): dataframe to split 

Returns
   qb_data, rb_data, te_data, wr_data (tuple): split dataframes by position
'''
def split_data_by_position(df: pd.DataFrame): 
   # split data by position
   qb_data = df[df['position'] == 'QB']
   rb_data = df[df['position'] == 'RB']
   wr_data = df[df['position'] == 'WR']
   te_data = df[df['position'] == 'TE']
   
   # drop un-needed position column
   new_qb_data = qb_data.drop('position', axis=1)
   new_rb_data = rb_data.drop('position', axis=1)
   new_wr_data = wr_data.drop('position', axis=1)
   new_te_data = te_data.drop('position', axis=1)
   
   return get_rankings_ratios(new_qb_data, new_rb_data, new_wr_data, new_te_data)
   
   

'''
Filter out outliers from data 

Args:
   df (pd.DataFrame): dataframe to filter out records 

Returns:
   filtered_df (pd.DataFrame): data frame with records filtered
'''
def filter_data(df: pd.DataFrame):
   df = df.dropna() 
   non_zero_data = df[(df['fantasy_points'] > 0) & (df['avg_fantasy_points'] > 5)] 
   
   
   upper_fantasy_points_outliers = non_zero_data['fantasy_points'].quantile(.99)
   lower_fantasy_points_outliers = non_zero_data['fantasy_points'].quantile(.01)
   
   # remove top 99% and bottom 1% of fantasy points
   no_fantasy_points_outliers = non_zero_data[(non_zero_data['fantasy_points'] < upper_fantasy_points_outliers) & (non_zero_data['fantasy_points'] > lower_fantasy_points_outliers)]
   
   upper_avg_fantasy_points_outliers = no_fantasy_points_outliers['avg_fantasy_points'].quantile(.99)
   lower_avg_fantasy_points_outliers = no_fantasy_points_outliers['avg_fantasy_points'].quantile(.01)
   
   # remove top 99% and bottom 1% of avg_fantasy_points
   no_outlier_data = no_fantasy_points_outliers[(no_fantasy_points_outliers['avg_fantasy_points'] < upper_avg_fantasy_points_outliers) & (no_fantasy_points_outliers['avg_fantasy_points'] > lower_avg_fantasy_points_outliers)]
   
   return no_outlier_data


'''
Remove any skewed nature from our features 

Args:
   qb_data (pd.DataFrame): qb dataframe to transform 
   rb_data (pd.DataFrame): rb dataframe to transform 
   wr_data (pd.DataFrame): wr dataframe to transform 
   te_data (pd.DataFrame): te dataframe to transform 
Returns:
   None
'''
def transform_data(qb_data: pd.DataFrame, rb_data: pd.DataFrame, wr_data: pd.DataFrame, te_data: pd.DataFrame):
   for df in [qb_data, rb_data, wr_data, te_data]:
      logged_avg_fantasy_points = np.log1p(df['avg_fantasy_points'])
      logged_fantasy_points = np.log1p(df['fantasy_points'])
      
      if 'rush_ratio_rank' in df.columns:
         logged_ratio_rank = np.log1p(df['rush_ratio_rank'])
      else:
         logged_ratio_rank = np.log1p(df['pass_ratio_rank'])
         
      df['log_avg_fantasy_points'] = logged_avg_fantasy_points
      df['log_fantasy_points'] = logged_fantasy_points
      df['log_ratio_rank'] = logged_ratio_rank 
   
   
   return qb_data, rb_data, wr_data, te_data


'''
Calculate ratio of relevant defensive ranking to relevant offensive ranking 

Args:
   df (pd.DataFrame): dataframe to generate relevant ranks for 
   is_rushing (bool): boolean to determine if we are calculating this for rushing or for passing

Returns:
   updated_df (pd.DataFrame): dataframe with correct rank ratios
'''
def get_rankings_ratios(qb_data: pd.DataFrame, rb_data: pd.DataFrame, wr_data: pd.DataFrame, te_data: pd.DataFrame): 
   rush_ratio_rank = rb_data['def_rush_rank'] / rb_data['off_rush_rank']
   rb_data['rush_ratio_rank'] = rush_ratio_rank
   
   for df in [qb_data, wr_data, te_data]:
      pass_ratio_rank = df['def_pass_rank'] / df['off_pass_rank']
      df['pass_ratio_rank'] = pass_ratio_rank

   return qb_data, rb_data, wr_data, te_data
   

'''
Addition of avg_fantasy_points feature so historical data is included in model 

Args:
   df (pd.DataFrame): dataframe to add avg_fantasy_points to 

Returns:
   df (pd.DataFrame): updated data frame 
'''
def include_averages(df: pd.DataFrame):
   player_avg_points = df.groupby('player_id')['fantasy_points'].mean()
   df['avg_fantasy_points'] = df['player_id'].map(player_avg_points)
   return df


'''
Plot a series in order to determine if outliers exists 

Args:
   series (pd.Series): series to plot 
   pdf_name (str): file name for pdf

Returns:
   None
'''
def plot(series: pd.Series, pdf_name: str):
   sns.displot(series, kind="hist", kde=True, bins=10, 
            color="skyblue", edgecolor="white")
   
   relative_dir = "./data/distributions"
   file_name = f"{pdf_name}.pdf"
   os.makedirs(relative_dir, exist_ok=True)
   file_path = os.path.join(relative_dir, file_name)
   
   plt.savefig(file_path)
   plt.close()
   

'''
Create scatter plots for features vs dependent variable

Args:
   data (pd.DataFrame): dataframe to yank tdata from 
   output_file (str): file name

Returns:
   None
'''
def create_plots(data: pd.DataFrame, position: str):
   f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(15,6))
   ax1.scatter(data['log_avg_fantasy_points'],data['log_fantasy_points'])
   ax1.set_title('Log Fantasy Points and Log Avg Fantasy Points')
   
   ax2.scatter(data['log_ratio_rank'],data['log_fantasy_points'])
   ax2.set_title('Log Fantasy Points and Log Ratio Rank')

   relative_dir = "./data/scatter"
   file_name = f"{position}_features_plot.pdf"
   os.makedirs(relative_dir, exist_ok=True)
   file_path = os.path.join(relative_dir, file_name)
   f.tight_layout()
   
   plt.savefig(file_path)



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
