import pandas as pd
from db import fetch_data as fetch



def preprocess(): 
   df = fetch_data()
   print(f"Original Data: {len(df)} rows")  

   parsed_df = parse_player_props(df)

   processed_df = pd.get_dummies(parsed_df, columns=['position'], dtype=int) #encode categoricla variable
   #TODO: Drop the week column as well since this is NOT a independent variable
   processed_df.drop(columns=['player_id'], inplace=True) # drop un-needed values 

   return processed_df
   


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
   return fetch.fetch_independent_and_dependent_variables()