import pandas as pd
from db import insert_data
from config import props
import time


def ingest_data_sets(start_year: int, end_year: int):
    """Functionality to ingest data sets and insert relevant records into our database

    Args:
        data_set_name (str): the name corresponding to the dataset 
        start_year (int): the year to start fetching metrics for 
        end_year (int): the year to stop fetching metrics for 
    """
    teams = [
        {
            "team": team["name"],
            "acronym": team.get("our_lads_acronym") or team.get("rotowire") or team["pfr_acronym"]
        }
        for team in props.get_config("nfl.teams")
    ]
    # update edge cases
    for team in teams:
        if team["team"] == "Los Angeles Rams":
            team["acronym"] = "LA"  
        elif team["team"] == "Arizona Cardinals":
            team["acronym"] = "ARI"  


    rel_dir = "../datasets"
    data_set_names = ['weekly_player_data.csv', 'weekly_team_data.csv', 'yearly_player_data.csv', 'yearly_team_data.csv']

    for dataset in data_set_names: 
        df = pd.read_csv(f"{rel_dir}/{dataset}")
        relevant_data = df[(df["season"] >= start_year) & (df["season"] <= end_year)]

        # insert yearly team data
        if dataset == "yearly_team_data.csv":
            insert_data.insert_team_seasonal_general_metrics(relevant_data, teams)
            insert_data.insert_team_seasonal_passing_metrics(relevant_data, teams)
            insert_data.insert_team_seasonal_rushing_metrics(relevant_data, teams)
        

        # insert yearly player data
        if dataset == "yearly_player_data.csv":
            for year in range(start_year, end_year + 1):
                print(f"Attmpeintg to insert player demographics for year {year}")
                insert_data.insert_player_demographics(relevant_data[relevant_data["season"] == year])

                #TODO: Account for other metrics 




