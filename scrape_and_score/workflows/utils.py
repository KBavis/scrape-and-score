from db import fetch_data


def is_first_execution_of_season(season: int): 
    """
    Utility function to determine if our application has been run for particular season and has relevant metrics perisisted

    Args:
        seaon (int): the season to check for 
    
    Returns:
        bool: flag indicating if necessary data still needs to be persisted
    """ 

    # player demographics
    num_pd_records = fetch_data.get_count_player_demographics_records_for_season(season)

    # player teams 
    num_pt_records = fetch_data.get_count_player_teams_records_for_season(season) #NOTE: If player_teams is empty, relevant players / depth_chart_position records are also assumed to be lacking for specific season since this is done in a single flow

    return num_pd_records == 0 and num_pt_records == 0