import pandas as pd
from db import fetch_data as fetch
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import logging
import re
from constants import QB_FEATURES, RB_FEATURES, WR_FEATURES, TE_FEATURES
import time

# global variable to account for dynamic categorical column names
injury_feature_names = []

def preprocess(): 
   logging.info('Attempting to fetch & pre-process our training/testing data for our Neural Networks...')
   df = fetch_data()

   parsed_df = parse_player_props(df)
   parsed_df = encode_player_injuries(parsed_df)
   parsed_df = encode_game_conditions(parsed_df)

   processed_df = pd.get_dummies(parsed_df, columns=['position'], dtype=int) #encode categorical variable
   processed_df.drop(columns=['player_id'], inplace=True) # drop un-needed values 

   # independent variables to account for cyclical nature
   cyclical_df = processed_df[['week', 'season']].copy()
   cyclical_df['max_week'] = cyclical_df['season'].apply(lambda season: 17 if season < 2021 else 18)
   cyclical_df['week_cos'] = np.cos(2 * np.pi * cyclical_df['week'] / cyclical_df['max_week'])
   cyclical_df['week_sin'] = np.sin(2 * np.pi * cyclical_df['week'] / cyclical_df['max_week'])
   cyclical_df = cyclical_df[['week_cos', 'week_sin']] # only keep transformed features
   
   # add cyclical features to preprocessed data frame
   processed_df[['week_cos', 'week_sin']] = cyclical_df

   #TODO: Ensure that features that -1 makes sense for are udpated to use a differnt value (i.,e -100)
   processed_df.fillna(-1, inplace=True) # fill remaining NA values with -1

   return processed_df


def feature_selection(df: pd.DataFrame, position: str):
   """
   Helper function to select relevant features for training & testing 

   Args:
      df (pd.DataFrame): data frame to perform feature selection for 
      position (str): the position relating to the data we are performing feature selection on 
   """
   logging.info(f"Attempting to determine relevant features for our {position} Neural Network Model")

   # extract our inputs/outputs
   Xs, inputs = scale_and_transform(df, True)
   y = df['fantasy_points']

   # utilize LassoCV to determine relevant features
   lasso = LassoCV(cv=5, max_iter=100000).fit(Xs, y)
   selector = SelectFromModel(lasso, prefit=True)
   selector.transform(Xs)

   inputs_df = pd.DataFrame(data=inputs, columns=['feature'])
   selected_features = inputs_df[selector.get_support()]

   features = []
   for _,row in selected_features.iterrows():
      # append relevant features to include 
      features.append(row['feature'])

   # manual selection of features 
   feature_mapping = {"QB": QB_FEATURES, "RB": RB_FEATURES, "TE": TE_FEATURES, "WR": WR_FEATURES}
   for feature in feature_mapping.get(position) : 
      if feature not in features:
         logging.info(f'Manually adding the following feature to our {position} Neural Network Model: {feature}')
         features.append(feature)

   logging.info(f'Selected Features via LassoCV and Manual Feature Selection: \n\n\t{features}') 
   return features

   

def scale_and_transform(df: pd.DataFrame, return_inputs: bool = False):
   """
   Functionality to scale and transform data frame and return respective inputs / ouputs 

   Args:
      df (pd.DataFrame): data frame containing X's & Y's 
      return_inputs (bool): flag to determine if we should return inputs corresponding to scaled values
   
   Returns:
      np.array: numpy array containing scaled inputs 
   """
   logging.info("Scaling and transforming DF in order to utilize in Neural Network training & testing")

   global injury_feature_names
   scaler = StandardScaler()

   # store independent variables in seperate data frame 
   xs = df.drop(columns=['fantasy_points']).copy()

   # extract game condition hot encoded features
   columns = df.columns
   game_condition_features = ['month']
   for column in columns:
      if column.startswith('precip_type') or column.startswith('weather_status') or column.startswith('surface'):
         game_condition_features.append(column)

   # independent variables to avoid scaling due to categorical nature
   position_columns = ['position_QB', 'position_RB', 'position_WR', 'position_TE']
   categorical_columns = [
      'wednesday_practice_status',
      'thursday_practice_status',
      'friday_practice_status',
      'official_game_status'
   ] + injury_feature_names + game_condition_features


   # extract columns that aren't present in df
   valid_positions = [col for col in position_columns if col in xs.columns]
   valid_categoricals = [col for col in categorical_columns if col in xs.columns]

   # independent variables accounting for categorical values
   categorical_df = xs[valid_positions + valid_categoricals].copy()
   categorical_vals = categorical_df.values

   # independent variables to account for cyclical nature
   cyclical_df = xs[['week_cos', 'week_sin']]
   cyclical_week_vals = cyclical_df.values

   # drop un-needed columns in original indep. variables 
   columns_to_drop = valid_positions + valid_categoricals + ['week', 'season', 'week_cos', 'week_sin']
   valid_cols_to_drop = [col for col in columns_to_drop if col in columns]
   xs = xs.drop(columns=valid_cols_to_drop)


   # log out situations where there are any duplicate feautres  
   cols = list(xs.columns) + list(categorical_df.columns) + list(cyclical_df.columns)
   seen_cols = set()
   for col in cols:
      if col not in seen_cols:
         seen_cols.add(col)
      else:
         logging.warning(f"The following column has already been added: {col}")

   scaled_x_vals = scaler.fit_transform(xs.values)
   
   Xs = np.concatenate((scaled_x_vals, categorical_vals, cyclical_week_vals), axis=1)
   logging.info(f"Xs shape: {Xs.shape}")

   # determine if we should return inputs 
   if return_inputs is False:
      logging.info('return_inputs flag is false: Only returning scaled values')
      return Xs
   else:
      inputs = list(xs.columns) + categorical_columns + list(cyclical_df.columns)
      logging.info('return_inputs flag is true: Returning scaled values & transformed inputs')
      return Xs, inputs
   


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


def encode_game_conditions(df: pd.DataFrame) -> pd.DataFrame:
   """
   Encode game condition features via ordinal and one-hot encodings 

   Args:
       df (pd.DataFrame): data frame to encode 

   Returns:
       pd.DataFrame: data frame with enocded game conditions 
   """

   # encode months 
   month_mapping = {
      "January": 1, "February": 2, "March": 3, "April": 4,
      "May": 5, "June": 6, "July": 7, "August": 8,
      "September": 9, "October": 10, "November": 11, "December": 12
   }
   df['month'] = df['month'].map(month_mapping)

   # encode categorical features 
   categorical_features = ['surface', 'weather_status', 'precip_type']
   df['precip_type'] = df['precip_type'].replace('', None).fillna('unknown').map(normalize_game_condition)
   df['surface'] = df['surface'].replace('', None).fillna('unknown').map(normalize_game_condition)
   df['weather_status'] = df['weather_status'].replace('', None).fillna('unknown').map(normalize_game_condition)
   df = pd.get_dummies(df, columns=categorical_features, dtype=int)

   # fill na values for temperature (since -1 is a valid number for temperature)
   df['temperature'] = df['temperature'].fillna(-999)

   # remove percent sign from percip probability
   df['precip_probability'] = df['precip_probability'].replace('', None).fillna(-1)
   df['precip_probability'] = df['precip_probability'].str.replace('%', '', regex=False).astype(float)

   return df


def normalize_game_condition(entry: str):
   if not isinstance(entry, str) or not entry.strip():
      raise Exception(f'Unable to normalize game condition: {entry}')
   
   return entry.strip().lower().replace(' ', '_')
   


def encode_player_injuries(df: pd.DataFrame) -> pd.DataFrame:
   """
   Encode player injury features by converting statuses / injury locations into numerical values 

   Args:
       df (pd.DataFrame): data frame to encode player injuries for 

   Returns:
       pd.DataFrame: data frame with encoded features 
   """
   logging.info("Encoding player injuries with relevant numerical values instead of statuses / strings")

   global injury_feature_names

   # apply multi lable binaarizer to encode injury locations 
   df['injury_locations'] = df['injury_locations'].apply(preprocess_injury_locations)

   #TODO: The multi label binarizer currently adds a BUNCH of features, but we should remove this logic if doesn't add predictive power (70+ feautres added)
   mlb = MultiLabelBinarizer() 
   injury_locations_encoded = mlb.fit_transform(df['injury_locations'])

   new_cols = [ 'injury_' + col for col in mlb.classes_ ]
   injury_encoded_df = pd.DataFrame(injury_locations_encoded, columns=new_cols)

   # account for injury feature names dynamically 
   injury_feature_names = list(new_cols)

   df.drop(columns=['injury_locations'], inplace=True)
   df = pd.concat([df, injury_encoded_df], axis=1)


   # encoded practice statuses 
   practice_status_mapping = {
      'dnp': 0, 
      'limited': 1,
      'full': 2
   }
   df['wednesday_practice_status'] = df['wednesday_practice_status'].fillna('full').map(practice_status_mapping)
   df['thursday_practice_status'] = df['thursday_practice_status'].fillna('full').map(practice_status_mapping)
   df['friday_practice_status'] = df['friday_practice_status'].fillna('full').map(practice_status_mapping)

   # encode official game statuses
   official_game_status_mapping = {
      'out': 0,
      'doubtful': 1,
      'questionable': 2, 
      'healthy': 3
   }
   df['official_game_status'] = df['official_game_status'].fillna('healthy').map(official_game_status_mapping)

   return df


def preprocess_injury_locations(entry):
   """
   Preprocess injury locations entry to normalize and transform into list

   Args:
       entry (str): injury location to preprocess

   Returns:
       list : injury locations with white space remove and in list format
   """

   if not isinstance(entry, str) or not entry.strip():
      return ['na']

   # lowercase, remove extra spaces, and standardize separators
   entry = entry.lower()
   entry = re.sub(r'[,/]', ',', entry)  # convert slashes to commas
   entry = re.sub(r'\s*,\s*', ',', entry)  # strip around commas
   entry = re.sub(r'\s+', ' ', entry).strip()  # clean up extra spaces

   injuries = entry.split(',')

   return normalize_injury_locations(injuries)


def normalize_injury_locations(injuries: list):
   """
   Normalize injury locations in order to reduce the number of potential options and correlate related locations 

   Args:
      injuries (list): list of injury locations corresponding to a player 
   """

   # manual mappings of synonomous injury locations
   manual_mapping = {
      'hips': 'hip',
      'ribs': 'rib',
      'feet': 'foot',
      'ankles': 'ankle',
      'quadricep': 'quadriceps',
      'concussion_protocol': 'concussion',
   }

   normalized_injuries = []
   for injury in injuries:
      injury = injury.lower()

      # remove spaces & replace with underscore
      injury = injury.replace(' ', '_')

   
      # account for plurla/singular 
      if injury in manual_mapping:
         normalized_injuries.append(manual_mapping[injury])
         continue
      
      # account for non injury related injuries
      if 'not_injury_related' in injury or 'resting_vet' in injury:
         normalized_injuries.append('not_injury_related')
         continue

      # account for 'illness' related injuries 
      if 'ill' in injury or 'covid' in injury:
         normalized_injuries.append('illness')
         continue

      # account for niche scenarios that we have seen 
      if '_(rt.' in injury:
         normalized_injuries.append(injury.replace('_(rt.', ''))
         continue
      
      if ':' in injury:
         injury = injury.split(':')[-1]
         normalized_injuries.append(injury.strip('_'))
         continue


      # append normal injury if no specific updates required 
      normalized_injuries.append(injury)
   
   return normalized_injuries






def fetch_data(): 
   """
      Retrieve indepdenent variables 
   """
   logging.info("Fetching inputs & outputs for Neural Network training & testing")
   return fetch.fetch_independent_and_dependent_variables()