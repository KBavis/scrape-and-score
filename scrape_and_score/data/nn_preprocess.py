import pandas as pd
from db import fetch_data as fetch
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging



def preprocess(): 
   df = fetch_data()

   parsed_df = parse_player_props(df)

   processed_df = pd.get_dummies(parsed_df, columns=['position'], dtype=int) #encode categorical variable
   processed_df.drop(columns=['player_id'], inplace=True) # drop un-needed values 

   processed_df.fillna(-1, inplace=True) # fill remaining NA values with -1

   return processed_df
   

def scale_and_transform(df: pd.DataFrame):
   """
   Functionality to scale and transform data frame and return respective inputs / ouputs 

   Args:
      df (pd.DataFrame): data frame containing X's & Y's 
   
   Returns:
      np.array: numpy array containing scaled inputs 
   """
   logging.info("Scaling and transforming DF in order to utilize in Neural Network training & testing")
   scaler = StandardScaler()

   # store independent variables in seperate data frame 
   xs = df.drop(columns=['fantasy_points']).copy()

   # independent variables to avoid scaling due to categorical nature
   position_columns = ['position_QB', 'position_RB', 'position_WR', 'position_TE']
   valid_positions = [] 
   for position in position_columns: 
      if position in xs.columns: 
         valid_positions.append(position)
      
   categorical_df = xs[valid_positions].copy()
   categorical_vals = categorical_df.values

   # independent variables to account for cyclical nature
   cyclical_df = xs[['week', 'season']].copy()

   cyclical_df['max_week'] = cyclical_df['season'].apply(lambda season: 17 if season < 2021 else 18)
   cyclical_df['week_cos'] = np.cos(2 * np.pi * cyclical_df['week'] / cyclical_df['max_week'])
   cyclical_df['week_sin'] = np.sin(2 * np.pi * cyclical_df['week'] / cyclical_df['max_week'])
   cyclical_df = cyclical_df[['week_cos', 'week_sin']] # only keep transformed features
   cyclical_week_vals = cyclical_df.values

   # drop un-needed columns in original indep. variables 
   columns_to_drop = valid_positions + ['week', 'season']
   xs = xs.drop(columns=columns_to_drop)

   scaled_x_vals = scaler.fit_transform(xs.values)
   
   Xs = np.concatenate((scaled_x_vals, categorical_vals, cyclical_week_vals), axis=1)
   logging.info(f"Xs shape: {Xs.shape}")
   return Xs
   


def parse_player_props(df: pd.DataFrame):
   """Parse out player props retrieved from database 

   Args:
       df (pd.DataFrame): data frame containing relevant independent & dependent variables 

   Returns:
       df (pd.DataFrame): data frame updated with relevant player prop columns 
   """
   
   parsed_data = []

   for _, row in df.iterrows():
      week_props = row["props"]

      row_data = {}

      for prop in week_props:
         label = prop["label"].lower().replace(" ", "_").replace("/", "_")
         
         #TODO: Remove this logic as there no longer will be any (over) or (under) labels
         if "(under)" in label:
            break # skip under lines (only account for over for simplicity)
         else :
            label = label.replace("_(over)", "") # remove over indicator since all lines/costs are over 
         
         cost = prop["cost"]
         line = prop["line"]

         row_data[f"{label}_cost"] = cost
         row_data[f"{label}_line"] = line

      parsed_data.append(row_data)

   parsed_df = pd.DataFrame(parsed_data, index=df.index)
   parsed_df.fillna(-1, inplace=True)
   
   df = df.drop(columns=["props"])
   return pd.concat([df, parsed_df], axis=1)
   


def fetch_data(): 
   """
      Retrieve indepdenent variables 
   """
   logging.info("Fetching inputs & outputs for Neural Network training & testing")
   return fetch.fetch_independent_and_dependent_variables()