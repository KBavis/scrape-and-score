from db import fetch_data
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging


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
   
   transformed_data = transform_data(filtered_df)
   logging.info(transformed_data.describe(include='all'))
   
   plot(transformed_data['log_fantasy_points'], 'log_fantasy_points_distribution')
   plot(transformed_data['log_avg_fantasy_points'], 'log_avg_fantasy_points_distribution')
   
   create_plots(transformed_data)
   

'''
Filter out outliers from data 

Args:
   df (pd.DataFrame): dataframe to filter out records 

Returns:
   filtered_df (pd.DataFrame): data frame with records filtered
'''
def filter_data(df: pd.DataFrame):
   df = df.dropna() 
   non_zero_data = df[(df['fantasy_points'] > 0) & (df['avg_fantasy_points'] > 3)] 
   
   
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
   df (pd.DataFrame): dataframe to transform 

Returns:
   None
'''
def transform_data(df: pd.DataFrame):
   logged_fantasy_points = np.log1p(df['fantasy_points'])
   df['log_fantasy_points'] = logged_fantasy_points
   
   logged_avg_fantasy_points = np.log1p(df['avg_fantasy_points'])
   df['log_avg_fantasy_points'] = logged_avg_fantasy_points
   return df
   

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
Plot a series in order to determine if outliers exists 

Args:
   series (pd.Series): series to plot 
   pdf_name (str): file name for pdf

Returns:
   None
'''
def plot(series: pd.Series, pdf_name: str):
   sns.displot(series)
   plt.savefig(f"{pdf_name}.pdf")
   plt.close()
   

'''
Create scatter plots for features vs dependent variable

Args:
   data (pd.DataFrame): dataframe to yank tdata from 
   output_file (str): file name

Returns:
   None
'''
def create_plots(data: pd.DataFrame, output_file: str = 'featureplots.pdf'):
   f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
   ax1.scatter(data['avg_fantasy_points'],data['log_fantasy_points'])
   ax1.set_title('Log Fantasy Points and Avg Fantasy Points')
   ax2.scatter(data['off_rush_rank'],data['log_fantasy_points'])
   ax2.set_title('Log Fantasy Points and Off Rush Rank')
   ax3.scatter(data['off_pass_rank'],data['log_fantasy_points'])
   ax3.set_title('Log Fantasy Points and Off Pass Rank')

   plt.savefig(output_file)

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
