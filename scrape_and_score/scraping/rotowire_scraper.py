import pandas as pd
import requests
import logging
from config import props
from service import team_service
from db import insert_data
from db import fetch_data

"""
Fetch all historical odds for the current year

Args:
   None 

Returns:
   None
"""


def scrape_all():
    # fetch configs
    url = props.get_config("website.rotowire.urls.historical-odds")
    curr_year = props.get_config("nfl.current-year")

    # retrieve historical odds
    jsonData = requests.get(url).json()
    df = pd.DataFrame(jsonData)

    # generate team betting odd records to persist for current year
    curr_year_data = df[df["season"] == str(curr_year)]
    team_betting_odds_records = get_team_betting_odds_records(curr_year_data)

    # insert into our db
    logging.info("Inserting all teams historical odds into our database")
    insert_data.insert_teams_odds(team_betting_odds_records)


"""
Generate team betting odds records to persist into our database

Args:
   curr_year_data (pd.DataFrame): betting odds corresponding to current year
"""


def get_team_betting_odds_records(curr_year_data: pd.DataFrame):
    # create team id mapping
    mapping = create_team_id_mapping(curr_year_data[curr_year_data["week"] == "1"])
    betting_odds_records = [
        {
            "home_team_id": mapping[row["home_team_stats_id"]],
            "away_team_id": mapping[row["visit_team_stats_id"]],
            "home_team_score": row["home_team_score"],
            "away_team_score": row["visit_team_score"],
            "week": row["week"],
            "year": row["season"],
            "game_over_under": row["game_over_under"],
            "favorite_team_id": mapping[row["favorite"]],
            "spread": row["spread"],
            "total_points": row["total"],
            "over_hit": row["over_hit"],
            "under_hit": row["under_hit"],
            "favorite_covered": row["favorite_covered"],
            "underdog_covered": row["underdog_covered"],
        }
        for _, row in curr_year_data.iterrows()
    ]
    return betting_odds_records


"""
Create a mapping of a teams acronym to corresponding team ID in our database

Args:
   week_one_data (pd.DataFrame): data corresponding to week one of current NFL season 

Returns:
   mapping (dict): mapping of a teams acronmy to team_id in our db
"""


def create_team_id_mapping(week_one_data: pd.DataFrame):
    # load configs
    teams = props.get_config("nfl.teams")
    mappings = {}

    for team in teams:
        acronym = team.get("rotowire", team["pfr_acronym"])
        mappings[acronym] = team_service.get_team_id_by_name(team["name"])

    return mappings


"""
Update persisted betting records with relevant information (under/over hit, coverd, total score, etc)

Args:
   None

Returns:
   None
"""


def update_recent_betting_records():
    # load configs
    url = props.get_config("website.rotowire.urls.historical-odds")
    curr_year = props.get_config("nfl.current-year")

    # determine recent week
    recent_week = fetch_data.fetch_max_week_persisted_in_team_betting_odds_table(
        curr_year
    )

    # retrieve historical odds
    jsonData = requests.get(url).json()
    df = pd.DataFrame(jsonData)
    print(df)

    # create ID mapping
    mapping = create_team_id_mapping(
        df[(df["week"] == "1") & (df["season"] == str(curr_year))]
    )
    print(mapping)

    # filter out records based on year & recent week
    recent_data = df[
        (df["season"] == str(curr_year)) & (df["week"] == str(recent_week))
    ]

    # insert updates
    records = generate_update_records(recent_data, mapping, curr_year, recent_week)
    insert_data.update_team_betting_odds_records_with_outcomes(records)


"""
Generate records to update 'team_betting_odds' records that games have been completed 

Args:
   recent_data (pd.DataFrame): dataframe containing team betting odds information from most recent week & year 
   mapping (dict) : mapping of a teams acronym to its corresponding team_id
   year (int): season the updated records correspond to 
   week (int): the week these records correspond to
   
Returns:
   None
"""


def generate_update_records(
    recent_data: pd.DataFrame, mapping: dict, year: int, week: int
):
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


"""
Fetch odds for upcoming game that is to be played so we can make predicitions 

Args:
   None 

Returns:
   None
"""


def scrape_upcoming():
    # extract configs
    url = props.get_config("website.rotowire.urls.upcoming-odds")
    curr_year = props.get_config("nfl.current-year")

    # extract latest week that we have bet data persisted
    week = fetch_data.fetch_max_week_persisted_in_team_betting_odds_table(curr_year)
    week_to_fetch = week + 1  # fetch data for next week

    # fetch upcoming odds
    params = {"week": week_to_fetch}
    jsonData = requests.get(url, params=params).json()
    df = pd.DataFrame(jsonData)

    # extract unique game IDs
    game_ids = df["gameID"].unique()
    logging.info(f"Unique game IDs representing upcoming NFL games: {game_ids}")

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
        df, game_ids, week_to_fetch, curr_year
    )

    insert_data.insert_teams_odds(upcoming_betting_odds, True)


"""
Generate upcoming betting odds records to persist to our DB 

Args:
   df (pd.DataFrame): data frame containing relevant metrics regarding upcoming game odds 
   game_ids (list): list of unique game IDs to generate records for 
   week (int): week corresponding to betting odds 
   year (int): season correspondign to betting odds 

Returns:
   records (list): list of betting odds to persist
"""


def get_upcoming_betting_odds_records(
    df: pd.DataFrame, game_ids: list, week: int, year: int
):
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

    return records


"""
Funcionality to retrieve the team ID of the favorite based on datafrmae game lines 

Args:
   game_df (pd.DataFrame): data frame containing team odds 

Returns:
   team_id (int): ID of the team who is favorited
"""


def get_favorite_team_id(game_df: pd.DataFrame):
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


"""
Calculate the avg O/U, avg spread, and avg_money line based on each available lines from relevant bookies 

Args:
   game_df (pd.DataFrame): data frame containing game betting lines for both home & away team

Returns:
   avg_ou, avg_spread (tuple): calculated avgs
"""


def calculate_avg_lines(game_df: pd.DataFrame):
    # list of possible bookies
    bookies = ["betrivers", "mgm", "draftkings", "fanduel", "best"]

    ou_count = 0
    ou_amount = 0
    spread_count = 0
    spread_amount = 0
    ml_count = 0
    ml_amount = 0

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
    avg_ml = ml_amount / ml_count if ml_count > 0 else None

    return avg_ou, avg_spread
