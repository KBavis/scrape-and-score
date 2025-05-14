from .util import fetch_page
from config import props
from bs4 import BeautifulSoup
from datetime import datetime
from db import fetch_data
import logging




def scrape_upcoming_games(season: int, week: int):
    """
    Functionality to scrape upcoming games from espon.com

    Args:
        season (int): season to extract games for 
        week (int): week to extract games for 
    """
    logging.info(f"Attempting to scrape upcoming Team Game Logs for Week {week} of the {season} NFL Season")

    url = props.get_config('website.espn.urls.upcoming').format(week, season)
    html = fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")

    week_game_days = soup.find_all('tbody', class_="Table__TBODY")
    game_dates = soup.find_all('div', class_='Table__Title')
    
    records = []
    for i, day in enumerate(week_game_days):
        games = day.find_all('tr')
        game_date = game_dates[i].text.strip()

        for game in games:

            home_team_cell = game.find('td',class_='events__col Table__TD')
            away_team_cell = game.find('td',class_='colspan__col Table__TD')

            home_team = extract_team_name(home_team_cell)
            away_team = extract_team_name(away_team_cell, is_away=True)

            records.append({"home_team": home_team, "away_team": away_team, "week": week, "season": season, "game_date": game_date})

    generate_and_persist(records, season, week)


def generate_and_persist(records: list, season: int, week: int): 
    """
    Generate and persist upcoming 'team_game_log' records

    Args:
        records (list): list of scraped records storing NFL game information 
        season (int): the season this corresponds to 
        week (int); the week this corresponds to 
    """

    logging.info(f"Attempting to generate and persist the following scraped records:\n\t{records}")

    mapping = generate_team_id_mapping()
    
    persistable_records = [] 
    for record in records:
        home_team_id = mapping[record['home_team']]
        away_team_id = mapping[record['away_team']]

        parsed_date = datetime.strptime(record['game_date'], "%A, %B %d, %Y")
        formatted_date = parsed_date.strftime("%m/%d/%Y")

        home_team_rest_days = calculate_rest_days(home_team_id, season, week, parsed_date)
        away_team_rest_days = calculate_rest_days(away_team_id, season, week, parsed_date)

        # generate two team game log records (one for each team) 
        persistable_records.append({"team_id": home_team_id, "opp": away_team_id, "is_home": True, "game_date": formatted_date, "rest_days": home_team_rest_days})
        persistable_records.append({"team_id": away_team_id, "opp": home_team_id, "is_home": False, "game_date": formatted_date, "rest_days": away_team_rest_days})
    

    filter(persistable_records)
    # TODO: Call insertion functionality after filtering (create new insert_upcoming_team_game_logs functionality in insert_data)


def calculate_rest_days(team_id: int, season: int, week: int, curr_game_date: datetime):

    # extract date from last weeks game 
    prev_game_date = fetch_data.fetch_game_date_from_team_game_log(season, week - 1, team_id)

    if prev_game_date is None:
        # attempt to extract week 18 match game date 
        prev_game_date = fetch_data.fetch_game_date_from_team_game_log(season - 1, 18, team_id)
        if prev_game_date is None:
            return 100  # default rest days to 100 if no previous date persisted. NOTE: This will be the case for all week 1 games of 2025 season. 
        
    

    # determine number of rest days since last game 
    rest_days = abs((curr_game_date - prev_game_date).days)
    return rest_days


def filter(records: list, season: int, week: int):
    """
    Functionality to filter out records that have been previosuly persisted and have no changes 

    Args:
        records (list): list of records to filter through 
        season (int): relevant season 
        week (int): relevant week 
    """


    for record in records:
        pk = {"team_id": record['team_id'], "week": week, "year": season}

        #TODO: This fetch_game-log_by_pk function currently is out dated. Update this function to specify individual attributes (ensuring we don't break any existing functionality in process) and ensure we include game date as well
        game_log = fetch_data.fetch_team_game_log_by_pk(pk) 

        if game_log:
            # check if modification occured to game date 
            # if changes occured, update team game log record in DB and filter out of records 
            # if no changes occured, log the below message
            logging.info("Team Game Log [team_id={},week={},season={}] already persisted & no updates required; skipping insertion")
        else:
            # no record exists, so we are free to insert
            continue


    
def generate_team_id_mapping():
    """
    Generate mapping of team's location to corresponding ID 

    Args:
        teams (list): list of persisted NFL teams 
    
    Returns:
        dict: mapping of a team name to a team ID 
    """

    teams = fetch_data.fetch_all_teams()
    return {record["name"].split(" ")[-1].lower(): record["team_id"] for record in teams}



def extract_team_name(team: BeautifulSoup, is_away: bool = False):
    """
    Extract team names from parsed HTML 

    Args:
        team (BeautifulSoup): parsed HTML 
        is_away (bool): flag indicating if this is for the away team or not 
    
    Returns:
        str: team's name
    """
    div = team.find_next('div')
    spans = div.find_all('span')

    if is_away:
        span = spans[1]
    else:
        span = spans[0]

    anchor_tags = span.find_all('a')
    a = anchor_tags[1]

    href = a.get('href')
    team_name = href.split("/")[-1]

    return team_name.split("-")[-1]
            

    
    










