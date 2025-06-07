from config import props
import requests
from service import player_service
from db.read.teams import (
    fetch_max_week_persisted_in_team_betting_odds_table,
    fetch_game_conditions_record_by_pk,
)
from db.read.players import (
    fetch_players_active_in_specified_year,
    fetch_player_betting_odds_record_by_pk,
)
from db.insert.players import (
    insert_player_props,
    insert_upcoming_player_props,
    update_upcoming_player_props,
)
from db.insert.teams import update_game_conditions, insert_game_conditions
import logging
from . import rotowire as rotowire
import time
from datetime import datetime
from constants import MARKET_ID_MAPPING


def fetch_historical_odds(season: int):
    """
    Fetch all historical player odds

    TODO (FFM-307): Add Concurrency

    Args:
        season (int): season to retrieve player odds for
    """

    max_week = fetch_max_week_persisted_in_team_betting_odds_table(season)
    markets = props.get_config("website.betting-pros.market-ids")

    players = fetch_players_active_in_specified_year(season)

    # iterate through each potential player
    # TODO (FFM-308): Account for player slugs for varying player slugs that aren't just first & last name
    for player in players:
        player_name = player["name"]
        logging.info(
            f'Fetching player props for the player "{player_name}" for the {season} season'
        )

        first_and_last = (
            player_name.replace("'", "").replace(".", "").lower().split(" ")[:2]
        )
        player_slug = "-".join(first_and_last)

        # iterate through each relevant week in specified season
        player_props = {"player_id": player["id"], "player_name": player_name}
        season_odds = []
        for week in range(1, max_week + 1):
            event_ids = fetch_event_ids_for_week(week, season)
            betting_odds = get_player_betting_odds(player_slug, event_ids, markets)

            if betting_odds == None:
                logging.info(
                    f"No betting odds retrieved for player {player_name} for week {week} in {season} season"
                )
                continue
            else:
                season_odds.append({"week": week, "week_odds": betting_odds})

        player_props.update({"season_odds": season_odds})

        # insert season long player props into db
        if season_odds:
            logging.info(
                f"Attempting to insert player props for player {player_name} for the {season} season..."
            )
            insert_player_props(player_props, season)
        else:
            logging.warn(
                f"No player props found for player {player_name} and season {season}; skipping insertion"
            )


def fetch_upcoming_player_odds_and_game_conditions(
    week: int, season: int, player_ids: list
):
    """
    Retrieve and persist upcoming player odds & game conditions

    Args:
        week (int): relevant week to retrieve data for
        season (int): reelvant season to retrieve data for
        player_ids (list): list of player IDs that we want to account for when scraping
    """

    # fetch & persist player odds
    fetch_upcoming_player_odds(week, season, player_ids)

    # fetch & persist game conditions
    fetch_upcoming_game_conditions(week, season)


def fetch_upcoming_game_conditions(week: int, season: int):
    """
    Retrieve upcoming game conditions & persist updates

    Args:
        week (int): relevant week
        season (int): relevant season
    """

    logging.info(
        f"Attempting to fetch upcoming 'game_conditions' records for week {week} of the {season} NFL seaons"
    )

    # extract games & their respective conditions for specific week/season
    url = props.get_config("website.betting-pros.urls.events")
    parsed_url = url.replace("{WEEK}", str(week)).replace("{YEAR}", str(season))
    json = get_data(parsed_url)

    records = []

    mapping = rotowire.create_team_id_mapping(True)

    # iterate through each event
    for event in json["events"]:

        # extract relevant stadium conditions
        surface = extract_surface(event["venue"])

        # extract relevant weather conditions
        weather = event["weather"]
        temperature = weather["forecast_temp"]
        wind_bearing = weather["forecast_wind_degree"]
        wind_speed = weather["forecast_wind_speed"]
        precip_prob = weather["forecast_rain_chance"]
        weather_status = weather["forecast_icon"]
        precip_type = extract_precip_type(weather["forecast_icon"])

        # extract relevant game time features
        game_date, game_time, kickoff, month, start = extract_game_time_metrics(
            event["scheduled"]
        )

        # extract home & away team ID
        home_id = mapping[event["home"]]
        away_id = mapping[event["visitor"]]

        records.append(
            {
                "surface": surface,
                "temperature": temperature,
                "wind_bearing": wind_bearing,
                "wind_speed": wind_speed,
                "precip_prob": precip_prob,
                "weather_status": weather_status,
                "precip_type": precip_type,
                "game_date": game_date,
                "game_time": game_time,
                "kickoff": kickoff,
                "month": month,
                "start": start,
                "season": season,
                "week": week,
                "home_team_id": home_id,
                "visit_team_id": away_id,
            }
        )

    # filter update / insert records
    update_records, insert_records = filter_game_conditions(records)

    # handle game_conditions insertions
    if insert_records:
        logging.info(
            f"Attempting to insert {len(insert_records)} game_conditions records for week {week} of the {season} NFL season"
        )
        insert_game_conditions(insert_records)
    else:
        logging.warning(
            f"No new game conditions retrieved for week {week} of the {season} NFL season; skipping insertion"
        )

    # handle game_conditions updates
    if update_records:
        logging.info(
            f"Attempting to update {len(update_records)} game_conditions records for week {week} of the {season} NFL season"
        )
        update_game_conditions(update_records)
    else:
        logging.warning(
            f"No updates to game conditions for week {week} of the {season} NFL season; skipping updates"
        )


def filter_game_conditions(records: list):
    """
    Filter upcoming 'game_conditions' records to either be insertable or updatable.

    Args:
        records (list): List of game condition records (scraped)

    Returns:
        tuple: (update_records, insert_records)
    """

    logging.info(
        "Attempting to filter 'game_conditions' into insertable vs updatable records"
    )

    update_records = []
    insert_records = []

    for record in records:
        pk = {
            "season": record["season"],
            "week": record["week"],
            "home_team_id": record["home_team_id"],
            "visit_team_id": record["visit_team_id"],
        }

        # Fetch existing record from DB
        persisted_record = fetch_game_conditions_record_by_pk(pk)

        if persisted_record is None:
            logging.info(
                f"No record persisted for PK={pk}; appending as insertable record"
            )
            insert_records.append(record)
            continue

        if are_game_conditions_modified(persisted_record, record):
            logging.info(
                f"Game conditions for PK={pk} have been modified; appending as updatable record"
            )
            update_records.append(record)

    return update_records, insert_records


def are_game_conditions_modified(persisted: dict, current: dict) -> bool:
    """
    Determines if any relevant game condition field has changed.

    Args:
        persisted (dict): The existing record from the DB
        current (dict): The newly scraped record

    Returns:
        bool: True if any tracked field has changed; otherwise False.
    """

    keys_to_compare = [
        "game_date",
        "game_time",
        "kickoff",
        "month",
        "start",
        "surface",
        "weather_icon",
        "temperature",
        "precip_probability",
        "precip_type",
        "wind_speed",
        "wind_bearing",
    ]

    for key in keys_to_compare:
        persisted_value = persisted.get(key)
        current_value = current.get(key)

        # account for different datatypes
        if key == "game_time":
            persisted_value = (
                int(persisted.get(key)) if persisted.get(key) is not None else None
            )
            current_value = (
                int(current.get(key)) if current.get(key) is not None else None
            )
        elif key == "temperature":
            persisted_value = (
                float(persisted.get(key)) if persisted.get(key) is not None else None
            )
            current_value = (
                float(current.get(key)) if current.get(key) is not None else None
            )

        if persisted_value != current_value:
            logging.info(
                f"The following 'game_condition' value has been modified: {key}"
            )
            print(f"Persisted Value: {persisted_value}, Current Value: {current_value}")
            return True
    return False


def extract_precip_type(weather_icon: str):
    """
    Extract the precip type from the weather icon

    Args:
        weather_icon (str): summation of current weather status

    Returns:
        precip_type (str): the precipitation type
    """

    if not weather_icon:
        return None

    weather = weather_icon.lower()

    if "rain" in weather:
        return "Rain"
    elif "snow" in weather:
        return "Snow"
    elif "sleet" in weather:
        return "Sleet"
    elif "hail" in weather:
        return "Hail"
    else:
        return None


def extract_game_time_metrics(datetime_str: str):
    """
    Extract relevant game time metrics from the scheduled kick off datetime

    Args:
        datetime_str (str): scheduled kickoff time in format "YYYY-MM-DD HH:MM:SS"

    Returns:
            tuple: (game_date, game_time, kickoff, month, start)
            - game_date: str, e.g., "2019-09-05 00:00:00"
            - game_time: int, e.g., 20 (hour in military time)
            - kickoff: str, e.g., "Sep 5 8:20 PM"
            - month: str, e.g., "September"
            - start: str, one of "Day", "Late", or "Night"
    """

    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

    # full datetime
    game_date = dt

    # military time of game
    game_time = dt.hour

    # kick off time
    kickoff = (
        dt.strftime("%b %-d %-I:%M %p")
        if hasattr(dt, "strftime")
        else dt.strftime("%b %#d %#I:%M %p")
    )

    # full month name
    month = dt.strftime("%B")

    # time of day classification
    if game_time < 12:
        start = "Day"
    elif 12 <= game_time < 17:
        start = "Late"
    else:
        start = "Night"

    return game_date, game_time, kickoff, month, start


def extract_surface(venue: dict):
    """
    Helper function to normalize the surface we are persisting
    TODO (FFM-309): Add mapping of stadium to stadium/surface type and split out surface vs stadium type

    Args:
        venue (dict): extracted venue from response

    Returns:
        str: normalized surface
    """

    if not venue:
        logging.warning(f"No 'venue' retrieved from GET:/events response")
        return ""

    if venue["stadium_type"] == "retractable_dome":
        return "Dome"

    if venue["surface"] == "turf" or venue["surface"] == "artificial":
        return "Turf"

    if "grass" in venue["surface"].lower():
        return "Grass"

    return venue["surface"].title() if venue["surface"] else ""


def fetch_upcoming_player_odds(week: int, season: int, player_ids: list):
    """
    Fetch relevant player props for upcoming NFL games & insert/update records into our DB

    TODO (FFM-310): Implement Retry Functionality

    Args:
        week (int): relevant week
        season (int): relevant season
        player_ids (list): relevant player IDs we need to fetch odds for
    """

    markets = props.get_config("website.betting-pros.market-ids")
    markets += ":253"  # account for projected fantasy points

    # extract event ids corresponding to week / season
    event_ids = fetch_event_ids_for_week(week, season)

    records = []

    # iterate through each potential player
    for player_id in player_ids:

        # extract player name by ID
        player_name = player_service.get_player_name_by_id(player_id)
        if player_name is None:
            logging.warning(
                f"Failed to retrieve player name corresponding to player ID {player_id}"
            )
            continue

        logging.info(
            f'Fetching player props for the player "{player_name}" for week {week} of the {season} NFL season'
        )

        # extract player slug for request
        first_and_last = (
            player_name.replace("'", "").replace(".", "").lower().split(" ")[:2]
        )
        player_slug = "-".join(first_and_last)

        # extract betting odds
        betting_odds = get_player_betting_odds(player_slug, event_ids, markets)

        if betting_odds == None:
            logging.info(
                f"No betting odds retrieved for player {player_name} for week {week} in {season} season"
            )
            continue

        records.append(
            {
                "week": week,
                "season": season,
                "player_id": player_id,
                "player_name": player_name,
                "odds": betting_odds,
            }
        )

    update_records, insert_records = filter_upcoming_player_odds(records)

    # insert player betting odds
    if insert_records:
        logging.info(
            f"Attempting to insert {len(insert_records)} player_betting_odds records for week {week} of the {season} NFL season"
        )
        insert_upcoming_player_props(insert_records)
    else:
        logging.warning(
            f"No new player props retrieved for week {week} of the {season} NFL season; skipping insertion"
        )

    # update player betting odds
    if update_records:
        logging.info(
            f"Attempting to update {len(update_records)}) player_betting_odds records for week {week} of the {season} NFL season"
        )
        update_upcoming_player_props(update_records)
    else:
        logging.warning(
            f"No lines/costs modifications for players betting lines for week {week} of the {season} NFL season; skipping updates"
        )


def filter_upcoming_player_odds(records: list):
    """
    Filter upcoming 'player_betting_odds' records to either be an insertable record or to be updated

    Args:
        records (list): list of records retrieved for players

    Returns:
        tuple: updateable & insertable records
    """

    logging.info(
        f"Attempting to filter upcoming 'player_betting_odds' into insertable records vs updateable records"
    )

    update_records = []
    insert_records = []

    # iterate through each indivudal player record
    for record in records:

        # extract relevant information
        player_id = record["player_id"]
        week = record["week"]
        season = record["season"]
        odds = record["odds"]

        # iterate through all available odds and filter
        for current_odds in odds:
            # extract label associated with current odds
            label = current_odds["label"]

            # generate current record corresponding to current odds
            curr_record = {
                "player_id": player_id,
                "player_name": record["player_name"],
                "week": week,
                "season": season,
                "label": label,
                "line": current_odds["line"],
                "cost": current_odds["cost"],
            }

            # retrieve player betting odds record by PK
            persisted_record = fetch_player_betting_odds_record_by_pk(
                player_id, week, season, label
            )
            if persisted_record is None:
                logging.info(
                    f"No record persisted for PK(player_id={player_id},week={week},season={season},label={label}); appending as insertable record"
                )
                insert_records.append(curr_record)
                continue

            # check if modifications were made to record
            if are_odds_modified(persisted_record, curr_record):
                logging.info(
                    f"Player '{record['player_name']}' player_betting_odds with PK(player_id={player_id},week={week},season={season},label={label}) has been modified; appending as updatable record"
                )
                update_records.append(curr_record)
                continue

    return update_records, insert_records


def are_odds_modified(persisted_record: dict, current_record: dict) -> bool:
    """
    Determines if the cost or line has changed for a player's odds record.

    Args:
        persisted_record (dict): The existing record from the DB
        current_record (dict): The new scraped record

    Returns:
        bool: True if either cost or line has changed; otherwise False.
    """

    # Check if cost or line has changed
    return (
        current_record["cost"] != persisted_record["cost"]
        or current_record["line"] != persisted_record["line"]
    )


def get_player_betting_odds(player_name: str, event_ids: str, market_ids: str):
    """
    Generate proprer URL needed to fetch relevant player prop metrics for a given week, player, and season 

    Args:
        player_name (str): player_slug to pass as a parameter to our request
        event_ids (str): all event_ids pertaining to the specified week 
        market_ids (str): all relevant market IDs to fetch odds for

    Returns:
        odds (dict): players odds 
    """
    base_url = props.get_config("website.betting-pros.urls.historical-odds")
    parsed_url = (
        base_url.replace("{MARKET_IDS}", market_ids)
        .replace("{PLAYER_SLUG}", player_name)
        .replace("{EVENT_IDS}", event_ids)
    )

    # fetch initial data from first page
    data = get_data(parsed_url.replace("{PAGE}", str(1)))

    if not data:
        return None

    num_pages = determine_number_of_pages(data)

    # no data for this player in the specified week
    if num_pages == 0:
        return None

    # account for odds on first page
    odds = get_odds(data, MARKET_ID_MAPPING)

    # loop through each possible page
    for page in range(2, num_pages + 1):
        data = get_data(parsed_url.replace("{PAGE}", str(page)))
        page_odds = get_odds(data, MARKET_ID_MAPPING)

        # account for additional odds available
        odds.extend(page_odds)

    # remove duplicates by keeping the one with the highest cost
    seen_labels = {}
    for odd in odds:
        label = odd["label"]
        if label in seen_labels:
            # if the label exists, compare costs and keep the higher cost
            if seen_labels[label]["cost"] < odd["cost"]:
                seen_labels[label] = odd
        else:
            seen_labels[label] = odd

    # return the filtered list of odds with duplicates removed
    filtered_odds = list(seen_labels.values())
    return filtered_odds


def determine_number_of_pages(data: dict):
    """
    Determine how many pages of data we should iterate through

    Args:
        data (dict): dictionary containing relevant player odds

    Returns:
        pages (int): # of pages to iterate through
    """

    if data and data.get("_pagination"):
        return data["_pagination"].get("total_pages", 0)
    return 0


def get_odds(data: dict, market_ids: dict):
    """
    Determine relevant odds based on API response

    Args:
        data (dict): relevant data from API response
        market_ids (dict): mapping of a market ID to corresponding props

    Returns:
        odds (dict): relevatn odds for player
    """

    odds = []

    # loop through available offers
    offers = data["offers"]
    for offer in offers:
        market_id = offer["market_id"]
        prop_name = market_ids[market_id]

        # loop through selections (will only be multiple if over/under)
        selections = offer["selections"]
        for selection in selections:

            # skip under lines, only account for overs
            if selection["label"] == "Under":
                continue

            if selection["label"] == "Over":
                full_prop_name = f"{prop_name} ({selection['label']})"
            else:
                full_prop_name = prop_name

            # loop through available bookies & find the best line
            bookies = selection["books"]
            best_found = False
            for book in bookies:
                # stop looping if we found best line
                if best_found:
                    break

                lines = book["lines"]

                # loop through available lines for a bookie
                for line in lines:
                    # only account for best lines
                    if line["best"] == True:
                        odds.append(
                            {
                                "label": full_prop_name,
                                "cost": line["cost"],
                                "line": line["line"],
                            }
                        )
                        best_found = True
                        break
    return odds


def get_data(url: str):
    """
    Retrieve JSON data from specified endpoint

    Args:
        url (str): url to retrieve data from

    Returns:
        jsonData (dict): json data
    """

    time.sleep(props.get_config("scraping.betting-pros.delay"))
    headers = {"x-api-key": props.get_config("website.betting-pros.api-key")}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        jsonData = response.json()
    except requests.RequestException as e:
        print(f"Error while fetching JSON content from the following URL {url} : {e}")
        return None

    return jsonData


def fetch_event_ids_for_week(week: int, year: int):
    """
    Retrieve event IDs (also known as game ID) for a given week/season

    Args:
        week (int): week to fetch ids for
        year (int): year to fetch ids for

    Returns:
        ids (str): delimited ids corresponding to particular week/year
    """

    url = props.get_config("website.betting-pros.urls.events")
    parsed_url = url.replace("{WEEK}", str(week)).replace("{YEAR}", str(year))

    json = get_data(parsed_url)

    ids = ":".join(str(event["id"]) for event in json["events"])
    return ids
