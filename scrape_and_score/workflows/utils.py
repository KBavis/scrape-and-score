from db import fetch_data


def is_player_demographics_persisted(season: int):
    """
    Utility function to determine if player demographics records are already persisted for certain season

    Args:
        seaon (int): the season to check for 
    
    Returns:
        bool: flag indicating if necessary data still needs to be persisted
    """ 
    # player demographics
    num_pd_records = fetch_data.get_count_player_demographics_records_for_season(season)

    return num_pd_records != 0

def is_player_records_persisted(season: int):
    """
    Utility function to determine if player_teams records are already persisted for certain season

    Args:
        seaon (int): the season to check for 
    
    Returns:
        bool: flag indicating if necessary data still needs to be persisted
    """ 
    # player teams 
    num_pt_records = fetch_data.get_count_player_teams_records_for_season(season) #NOTE: If player_teams is empty, relevant players / depth_chart_position records are also assumed to be lacking for specific season since this is done in a single flow

    return num_pt_records != 0