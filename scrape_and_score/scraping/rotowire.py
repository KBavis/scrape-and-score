import pandas as pd
import requests
import logging
from config import props
from service import team_service
from db.read.teams import (
    fetch_team_betting_odds_by_pk
)
from db.insert.teams import (
    insert_teams_odds,
    insert_game_conditions,
    update_team_betting_odds_records_with_outcomes,
    update_teams_odds
)



def scrape_all(start_year=None, end_year=None):
    """
    Fetch and persist all relevant team betting odds and game conditions spanning across multiple year
    """

    # fetch configs
    url = props.get_config("website.rotowire.urls.historical-odds")

    if start_year == None and end_year == None:
        curr_year = props.get_config("nfl.current-year") 

    # retrieve historical odds
    jsonData = requests.get(url).json()
    df = pd.DataFrame(jsonData)

    # generate team betting odd records to persist for current year
    data = df[df["season"] == str(curr_year)] if start_year == None and end_year == None else df[(df["season"] >= str(start_year)) & (df["season"] <= str(end_year))]


    team_betting_odds_records = get_team_betting_odds_records(data) 
    game_conditions = get_game_conditions(data)

    # insert into our db
    logging.info("Inserting all teams historical odds into our database")
    insert_teams_odds(team_betting_odds_records) 

    logging.info(f"Inserting game_conditions into our datbase from year {start_year} to year {end_year}")
    insert_game_conditions(game_conditions)




def get_game_conditions(data: pd.DataFrame) -> list:
    """
    Extract relevant game condition metrics from the API response and generate insertable records 

    Args:
        data (pd.DataFrame): data frame containing relevant game conditions 
    
    Returns:
        list : list of insertable game condition records
    """ 

    mapping = create_team_id_mapping()


    game_condition_records = [
        {
            "season": row['season'], 
            "week": row['week'],
            "home_team_id": mapping[row["home_team_stats_id"]],
            "visit_team_id": mapping[row["visit_team_stats_id"]],
            "game_date": row['game_date'], 
            "game_time": row['game_time'],
            "kickoff": row['kickoff'],
            "month": row['month'],
            "start": row['start'],
            "surface": row['surface'],
            "weather_icon": row['weather_icon'],
            "temperature": row['temperature'],
            "precip_probability": row['precip_probability'],
            "precip_type": row['precip_type'],
            "wind_speed": row['wind_speed'],
            "wind_bearing": row['wind_bearing']
        }
        for _, row in data.iterrows()
    ]

    return game_condition_records


def get_team_betting_odds_records(data: pd.DataFrame):
    """
    Generate team betting odds records to persist into our database

    Args:
    data (pd.DataFrame): betting odds corresponding to current year OR to a specified range of years
    """

    # create team id mapping
    mapping = create_team_id_mapping()

    betting_odds_records = [
    {
        "home_team_id": mapping[row["home_team_stats_id"]],
        "away_team_id": mapping[row["visit_team_stats_id"]],
        "home_team_score": row["home_team_score"],
        "away_team_score": row["visit_team_score"],
        "week": row["week"],
        "year": row["season"],
        "game_over_under": row["game_over_under"],
        "favorite_team_id": mapping[row["favorite"]] if row["favorite"] != "" else mapping[row["home_team_stats_id"]], #TODO (FFM-313): Update Impl to not automatically set home team as favorite if none present
        "spread": row["spread"],
        "total_points": row["total"],
        "over_hit": row["over_hit"],
        "under_hit": row["under_hit"],
        "favorite_covered": row["favorite_covered"],
        "underdog_covered": row["underdog_covered"],
    }
    for _, row in data.iterrows()
    ]

    return betting_odds_records 



def create_team_id_mapping(is_betting_pros: bool= False):
    """
    Create a mapping of a teams acronym to corresponding team ID in our database

    Args:
        is_betting_pros (bool): flag to indicate if we need ot use betting pros 

    Returns:
    mapping (dict): mapping of a teams acronmy to team_id in our db
    """

    # load configs
    teams = props.get_config("nfl.teams")
    mappings = {}

    for team in teams:
        if is_betting_pros:
            acronym = team.get("bettingpros") or team.get("rotowire") or team["pfr_acronym"]
        else:
            acronym = team.get("rotowire") or team["pfr_acronym"]
            
        alternate = None
        
        # account for alternate acronyms that have changed over year 
        if "alternate" in team: 
            alternate = team.get("alternate")

        mappings[acronym] = team_service.get_team_id_by_name(team["name"])

        # account for alternate acronym
        if alternate:
            mappings[alternate] = team_service.get_team_id_by_name(team["name"])

    return mappings



def update_recent_betting_records(week: int, season: int):
    """
    Update persisted betting records with relevant information (under/over hit, coverd, total score, etc)

    Args:
        week (int): the week to update records for 
        season (int): the season to update records for 
    """

    # load configs
    url = props.get_config("website.rotowire.urls.historical-odds")

    # retrieve historical odds
    jsonData = requests.get(url).json()
    df = pd.DataFrame(jsonData)
    print(df)

    # create ID mapping
    mapping = create_team_id_mapping(
        df[(df["week"] == "1") & (df["season"] == str(season))]
    )

    # filter out records based on year & recent week
    recent_data = df[
        (df["season"] == str(season)) & (df["week"] == str(week))
    ]

    # insert updates
    records = generate_update_records(recent_data, mapping, season, week)
    update_team_betting_odds_records_with_outcomes(records)



def generate_update_records(
    recent_data: pd.DataFrame, mapping: dict, year: int, week: int
):
    """
    Generate records to update 'team_betting_odds' records that games have been completed 

    Args:
        recent_data (pd.DataFrame): dataframe containing team betting odds information from most recent week & year 
        mapping (dict) : mapping of a teams acronym to its corresponding team_id
        year (int): season the updated records correspond to 
        week (int): the week these records correspond to
    """

    update_records = [
        {
            "home_team_id": mapping[row["home_team_stats_id"]],
            "visit_team_id": mapping[row["visit_team_stats_id"]],
            "total_points": row["total"],
            "over_hit": row["over_hit"],
            "under_hit": row["under_hit"],
            "favorite_covered": row["favorite_covered"],
            "underdog_covered": row["underdog_covered"],
            "home_team_score": row["home_team_score"],
            "visit_team_score": row["visit_team_score"],
            "year": year,
            "week": week,
        }
        for _, row in recent_data.iterrows()
    ]

    return update_records



def scrape_upcoming(week: int, season: int, team_ids: list = None):
    """
    Fetch odds for upcoming game that is to be played so we can make predicitions 
    """

    logging.info(
        f"Attempting to scrape upcoming 'team_betting_odds' for season={season}, week={week}"
        f"{f', team_ids={team_ids}' if team_ids is not None else ''}"
    )

    # extract configs
    url = props.get_config("website.rotowire.urls.upcoming-odds")

    # fetch upcoming odds
    params = {"week": week}
    jsonData = requests.get(url, params=params).json()
    df = pd.DataFrame(jsonData)

    # extract unique game IDs
    game_ids = df["gameID"].unique()

    # calculate avg o/u & spreads for each game
    df["avg_ou"] = None
    df["avg_spread"] = None

    for game_id in game_ids:
        game_betting_lines = df[df["gameID"] == game_id]

        if len(game_betting_lines) == 2:
            avg_ou, avg_spread = calculate_avg_lines(game_betting_lines)
            df.loc[df["gameID"] == game_id, "avg_ou"] = avg_ou
            df.loc[df["gameID"] == game_id, "avg_spread"] = avg_spread
        else:
            raise Exception(
                f"Unable to retrieve two unique records corresponding to game ID {game_id}"
            )

    # create persistable record s
    upcoming_betting_odds = get_upcoming_betting_odds_records(
        df, game_ids, week, season, team_ids
    )

    insert_records, update_records = filter_existing_records(upcoming_betting_odds)

    if insert_records:
        logging.info(f"Attempting to insert {len(insert_records)} 'team_betting_odds' records into DB")
        insert_teams_odds(upcoming_betting_odds, True)
    else:
        logging.info(f"No new 'team_betting_odds' records found for week {week} of the {season} NFL season; skipping insertion")
    
    if update_records:
        logging.info(f"Attempting to update {len(update_records)} 'team_betting_odds' records in DB")
        update_teams_odds(update_records)
    else:
        logging.info(f"No updates made to 'team_betting_odds' records for week {week} of the {season} NFL season; skipping updates")



def filter_existing_records(upcoming_betting_odds: list): 
    """
    Functionality to seperate our records that have already been persisted (and have changes) and records that should be inserted 

    Args:
        upcoming_betting_odds (list): list of upcoming betting odds to persist 

    Returns
        tuple: odds to update and odds to insert  
    """

    insert_records = []
    update_records = []

    for odds in upcoming_betting_odds:

        home_team_id = odds['home_team_id']
        away_team_id = odds['away_team_id']
        year = odds['year']
        week = odds['week']

        # check if record exists by PK 
        record = fetch_team_betting_odds_by_pk(home_team_id, away_team_id, year, week)
        if record is None:
            logging.info(f'No record persisted corresponding to PK (home_team_id={home_team_id},away_team_id={away_team_id},season={year},week={week}): Insertable record appended.')
            insert_records.append(odds)
            continue

        # check if any data has been modified 
        if are_odds_modified(record, odds):
            logging.info(f'Record persisted & updates are required for record with PK (home_team_id={home_team_id},away_team_id={away_team_id},season={year},week={week}): Updateable record appended.')
            update_records.append(odds)
            continue

        logging.info(f"Odds already persisted & no changes required for 'team_betting_odds' record with PK=(home_team_id={home_team_id},away_team_id={away_team_id},season={year},week={week})")


    return insert_records, update_records


def are_odds_modified(persisted_record: dict, current_record: dict):
    """"
    Helper function to determine if the odds have been modified 

    Args:
        persisted_record (dict): value persisted in DB 
        current_record (dict): value just scraped in this application invocation

    Returns:
        bool: flag indicating if we need to update record in DB 
    """

    return not (
        persisted_record["home_team_id"] == current_record["home_team_id"] and
        persisted_record["away_team_id"] == current_record["away_team_id"] and
        persisted_record["week"] == current_record["week"] and
        persisted_record["season"] == current_record["year"] and
        persisted_record["favorite_team_id"] == current_record["favorite_team_id"] and
        persisted_record["game_over_under"] == current_record["game_over_under"] and
        persisted_record["spread"] == current_record["spread"]
    )




def get_upcoming_betting_odds_records(
    df: pd.DataFrame, game_ids: list, week: int, year: int, team_ids: list 
):
    """
    Generate upcoming betting odds records to persist to our DB 

    Args:
        df (pd.DataFrame): data frame containing relevant metrics regarding upcoming game odds 
        game_ids (list): list of unique game IDs to generate records for 
        week (int): week corresponding to betting odds 
        year (int): season correspondign to betting odds 
        team_ids (list): relevant team IDs to acount for 

    Returns:
        records (list): list of betting odds to persist
    """

    logging.info(
        "Attempting to generate upcoming betting odds records to persist to our DB"
    )

    records = []

    for game_id in game_ids:
        game_df = df[df["gameID"] == game_id]

        if len(game_df) != 2:
            raise Exception("More than two records associated with game ID")

        # determine who is home vs away
        if game_df.iloc[0]["homeAway"] == "home":
            home_team_id = team_service.get_team_id_by_name(game_df.iloc[0]["name"])
            away_team_id = team_service.get_team_id_by_name(game_df.iloc[1]["name"])
        else:
            away_team_id = team_service.get_team_id_by_name(game_df.iloc[0]["name"])
            home_team_id = team_service.get_team_id_by_name(game_df.iloc[1]["name"])
        

        # check if away / home team ID is among relevant teams we want to fetch betting odds for 
        if team_ids is not None:
            if home_team_id not in team_ids or away_team_id not in team_ids:
                logging.info(f"The NFL Game taking place between Team {home_team_id} and Team {away_team_id} has already occurred; skipping fetching betting odds")
                continue

        favorite_team_id = get_favorite_team_id(game_df)

        # create & append record
        record = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "week": week,
            "year": year,
            "favorite_team_id": favorite_team_id,
            "game_over_under": df.iloc[0]["avg_ou"],
            "spread": df.iloc[0]["avg_spread"],
        }
        records.append(record)

    logging.info(f'Successfully generated {len(records)} upcoming team_betting_odds records')
    return records




def get_favorite_team_id(game_df: pd.DataFrame):
    """
    Funcionality to retrieve the team ID of the favorite based on datafrmae game lines 

    Args:
        game_df (pd.DataFrame): data frame containing team odds 

    Returns:
        team_id (int): ID of the team who is favorited
    """

    bookies = ["betrivers", "mgm", "draftkings", "fanduel", "best"]

    for bookie in bookies:
        has_ml = f"{bookie}_has_spread"
        ml = f"{bookie}_spread"

        if game_df.iloc[0][has_ml]:
            if float(game_df.iloc[0][ml]) > 0:
                return team_service.get_team_id_by_name(game_df.iloc[1]["name"])
            else:
                return team_service.get_team_id_by_name(game_df.iloc[0]["name"])

    raise Exception(
        "No bookies have available money lines; unable to determine favorited team"
    )


def calculate_avg_lines(game_df: pd.DataFrame):
    """
    Calculate the avg O/U and avg spread based on each available lines from relevant bookies 

    Args:
        game_df (pd.DataFrame): data frame containing game betting lines for both home & away team

    Returns:
        avg_ou, avg_spread (tuple): calculated avgs
    """

    # list of possible bookies
    bookies = ["betrivers", "mgm", "draftkings", "fanduel", "best"]

    ou_count = 0
    ou_amount = 0
    spread_count = 0
    spread_amount = 0

    for bookie in bookies:
        has_spread = f"{bookie}_has_spread"
        spread = f"{bookie}_spread"
        has_ou = f"{bookie}_has_ou"
        ou = f"{bookie}_ou"

        if (
            has_ou in game_df.columns
            and ou in game_df.columns
            and game_df.iloc[0].get(has_ou, False)
        ):
            ou_count += 1
            ou_amount += float(game_df.iloc[0][ou])

        if (
            has_spread in game_df.columns
            and spread in game_df.columns
            and game_df.iloc[0].get(has_spread, False)
        ):
            spread_count += 1
            spread_amount += float(game_df.iloc[0][spread])

        # Safely calculate averages to avoid division by zero
    avg_ou = ou_amount / ou_count if ou_count > 0 else None
    avg_spread = spread_amount / spread_count if spread_count > 0 else None

    return avg_ou, avg_spread
