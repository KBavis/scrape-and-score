import logging
import pandas as pd
from constants import TEAM_HREFS, MONTHS, LOCATIONS, CITIES, VALID_POSITIONS 
from config import props
from .util import fetch_page
from datetime import date, datetime
from bs4 import BeautifulSoup
from haversine import haversine, Unit
from rapidfuzz import fuzz


'''
Functionality to scrape relevant NFL teams and player data 

Args:
   team_and_player_data (list[dict]): every relevant fantasy NFL player corresponding to specified NFL season

Returns:
   data (tuple(list[pd.DataFrame], list[pd.DataFrame])) 
      - metrics for both players and teams
'''
def scrape(team_and_player_data: list):
   # TODO (FFM-31): Create logic to determine if new player/team data avaialable. If no new team data available, skip fetching metrics and utilize persisted metrics. If no new player data available, skip fetching metrics for player.
   
   # fetch configs 
   configs = props.load_configs()
   team_template_url = configs['website']['pro-football-reference']['urls']['team-metrics']
   year = configs['nfl']['current-year']
   
   # extract unique teams
   teams = {team['team'] for team in team_and_player_data}

   # fetch relevant team metrics 
   team_metrics = fetch_team_metrics(teams, team_template_url, year)
   
   # fetch relevant player metrics 
   player_metrics = fetch_player_metrics(team_and_player_data, year)
   
   # return metrics 
   return team_metrics, player_metrics 


'''
Functionality to fetch the metrics for each relevant player on current 53 man roster of specified year

Args:
   team_and_player_data (list[dict]) - every relevant fantasy NFL player corresponding to specified NFL season
   year (int) - year to fetch metrics for 
'''
def fetch_player_metrics(team_and_player_data, year):
   logging.info(f"Attempting to scrape player metrics for the year {year}")
    
   # order players by last name inital 
   ordered_players = order_players_by_last_name(team_and_player_data)
       
   # construct each players metrics link 
   player_urls = get_player_urls(ordered_players, year)
   
   # for each player url, fetch relevant metrics 
   for player_url in player_urls:
       url = player_url['url']
       player_name = player_url['player']
       
       raw_html = fetch_page(url)


'''
Functionality to fetch the metrics for each NFL team 

Args:
   teams (list) - list of team names to fetch metrics for 
   url_template (str) - template URL used to construct specific teams URL
   year (int) - year to fetch metrics for 
'''
def fetch_team_metrics(teams: list, url_template: str, year: int): 
   logging.info(f"Attempting to scrape team metrics for the following teams [{teams}]")
   
   team_metrics = []
   for team in teams: 
      logging.info(f"Fetching metrics for the following NFL Team: \'{team}\'")
      
      # fetch raw html for team 
      raw_html = get_team_metrics_html(team, year, url_template)
      
      # validate raw html fetched 
      if(raw_html == None):
         logging.error(f'An error occured while fetching raw HTML for the team \'{team}\'')
         raise Exception(f"Unable to extract raw HTML for the NFL Team \'{team}\'")
      
      # get team metrics from html 
      team_data = collect_team_data(team, raw_html, year)
      
      # validate teams metrics were retrieved properly 
      if(team_data.empty):
         logging.error(f'An error occured while fetching metrics for the team \'{team}\'')
         raise Exception(f"Unable to extract raw HTML for the NFL Team \'{team}\'")
      
      # append result 
      team_metrics.append({"team_name": team, "team_metrics": team_data})
   
   return team_metrics

'''
Functionality to fetch relevant metrics corresponding to a specific NFL team

All credit for the following code in this function goes to the developer of the repository:
      - https://github.com/mjk2244/pro-football-reference-web-scraper

Some subtle modifications were made to fix the repositories bug and to fit our use case.      
   
Args: 
    team (str) - NFL team full name
    raw_html (str) - raw HTML fetch for specified team
    year (int) - year to fetch metrics for 
      
Returns:
    pandas.DataFrame: A pandas DataFrame with relevant metrics corresponding to the specific player     
''' 
def collect_team_data(team: str, raw_html: str, year: int):
    
    #Configure Pandas DF 
    data = {
        'week': [],
        'day': [],
        'rest_days': [],
        'home_team': [],
        'distance_travelled': [],
        'opp': [],
        'result': [],
        'points_for': [],
        'points_allowed': [],
        'tot_yds': [],
        'pass_yds': [],
        'rush_yds': [],
        'opp_tot_yds': [],
        'opp_pass_yds': [],
        'opp_rush_yds': [],
    }
    df = pd.DataFrame(data)
    
    # create BeautifulSoup instance
    soup = BeautifulSoup(raw_html, "html.parser")

    #loading game data
    games = soup.find_all('tbody')[1].find_all('tr')

    # remove playoff games
    j = 0
    while j < len(games) and games[j].find('td', {'data-stat': 'game_date'}).text != 'Playoffs':
        j += 1
    for k in range(j, len(games)):
        games.pop()

    # remove bye weeks
    bye_weeks = []
    for j in range(len(games)):
        if games[j].find('td', {'data-stat': 'opp'}).text == 'Bye Week':
            bye_weeks.append(j)

    if len(bye_weeks) > 1:
        games.pop(bye_weeks[0])
        games.pop(bye_weeks[1] - 1)

    elif len(bye_weeks) == 1:
        games.pop(bye_weeks[0])

    # remove canceled games 
    to_delete = []
    for j in range(len(games)):
        if games[j].find('td', {'data-stat': 'boxscore_word'}).text == 'canceled':
            to_delete.append(j)
    for k in to_delete:
        games.pop(k)
    
    # remove games that have yet to be played  
    to_delete = []
    current_date = date.today()
    for j in range(len(games)):
        game_date = get_game_date(games[j], current_date)    
        if game_date >= current_date:
            to_delete.append(j)
    for k in reversed(to_delete): #reverse order to prevent shifting issue 
        games.pop(k)
      

    # gathering data for each game
    for i in range(len(games)):
        week = int(games[i].find('th', {'data-stat': 'week_num'}).text)
        day = games[i].find('td', {'data-stat': 'game_day_of_week'}).text
        if i > 0:
            if games[i - 1].find('td', {'data-stat': 'opp'}).text == 'Bye Week':
                date1 = games[i - 2].find('td', {'data-stat': 'game_date'}).text.split(' ')
                date2 = games[i].find('td', {'data-stat': 'game_date'}).text.split(' ')
            else:
                date1 = games[i - 1].find('td', {'data-stat': 'game_date'}).text.split(' ')
                date2 = games[i].find('td', {'data-stat': 'game_date'}).text.split(' ')
            if date1[0] == 'January':
                rest_days = date(year + 1, MONTHS[date2[0]], int(date2[1])) - date(
                    year + 1, MONTHS[date1[0]], int(date1[1])
                )
            elif date2[0] == 'January':
                rest_days = date(year + 1, MONTHS[date2[0]], int(date2[1])) - date(
                    year, MONTHS[date1[0]], int(date1[1])
                )
            else:
                rest_days = date(year + 1, MONTHS[date2[0]], int(date2[1])) - date(
                    year + 1, MONTHS[date1[0]], int(date1[1])
                )
        else:
            rest_days = date(2022, 7, 11) - date(2022, 7, 1)  # setting first game as 10 rest days

        opp = games[i].find('td', {'data-stat': 'opp'}).text

        if games[i].find('td', {'data-stat': 'game_location'}).text == '@':
            home_team = False
            distance_travelled = calculate_distance(LOCATIONS[CITIES[team]], LOCATIONS[CITIES[opp]])
        else:
            home_team = True
            distance_travelled = 0

        result = games[i].find('td', {'data-stat': 'game_outcome'}).text
        points_for = int(games[i].find('td', {'data-stat': 'pts_off'}).text)
        points_allowed = int(games[i].find('td', {'data-stat': 'pts_def'}).text)
        tot_yds = (
            int(games[i].find('td', {'data-stat': 'yards_off'}).text)
            if games[i].find('td', {'data-stat': 'yards_off'}).text != ''
            else 0
        )
        pass_yds = (
            int(games[i].find('td', {'data-stat': 'pass_yds_off'}).text)
            if games[i].find('td', {'data-stat': 'pass_yds_off'}).text != ''
            else 0
        )
        rush_yds = (
            int(games[i].find('td', {'data-stat': 'rush_yds_off'}).text)
            if games[i].find('td', {'data-stat': 'rush_yds_off'}).text != ''
            else 0
        )
        opp_tot_yds = (
            int(games[i].find('td', {'data-stat': 'yards_def'}).text)
            if games[i].find('td', {'data-stat': 'yards_def'}).text != ''
            else 0
        )
        opp_pass_yds = (
            int(games[i].find('td', {'data-stat': 'pass_yds_def'}).text)
            if games[i].find('td', {'data-stat': 'pass_yds_def'}).text != ''
            else 0
        )
        opp_rush_yds = (
            int(games[i].find('td', {'data-stat': 'rush_yds_def'}).text)
            if games[i].find('td', {'data-stat': 'pass_yds_def'}).text != ''
            else 0
        )

        # add row to data frame
        df.loc[len(df.index)] = [
            week,
            day,
            rest_days,
            home_team,
            distance_travelled,
            opp,
            result,
            points_for,
            points_allowed,
            tot_yds,
            pass_yds,
            rush_yds,
            opp_tot_yds,
            opp_pass_yds,
            opp_rush_yds,
        ]

    return df  


'''
Helper function to generate URL and fetch raw HTML for NFL Team

Args: 
    team_name (str) - NFL team full name
    year (int) - year to fetch raw HTML for
    url (str) - template URL to fetch HTML from
      
Returns:
    str - raw HTML from web page 
''' 
def get_team_metrics_html(team_name, year, url):
   url = url.replace("{TEAM_ACRONYM}", TEAM_HREFS[team_name]).replace("{CURRENT_YEAR}", str(year))
   return fetch_page(url)


'''
Functionality to calculate the distance between two cities 

All credit for the following code in this function goes to the developer of the repository:
      - https://github.com/mjk2244/pro-football-reference-web-scraper

   
Args: 
    city1 (dict) - dictionary containing a cities latitude & longitude 
    city2 (dict) - dictionary containing a cities latitude & longitude 
      
Returns:
    double: value corresponding to the distance between the two cities  

'''
def calculate_distance(city1: dict, city2: dict):
    coordinates1 = (city1['latitude'], city1['longitude'])
    coordinates2 = (city2['latitude'], city2['longitude'])
    return haversine(coordinates1, coordinates2, unit=Unit.MILES)


'''
Functionality to fetch the game date for a game 

Args:
    game (BeautifulSoup): BeautifulSoup object containing relevant game data 
    current_date (date) : the current date the application is being run 

Returns:
    game_date (date) : The date corresponding to the game day    
'''
def get_game_date(game: BeautifulSoup, current_date: date): 
    game_date_str = game.find('td', {'data-stat': 'game_date'}).text
    month_date = datetime.strptime(game_date_str, "%B %d").date() #convert to date time object
    if month_date.month == 1: 
        game_year = current_date.year + 1 #date corresponds to next year [TODO: determine how to fix this logic when we run this applicaiton in January?]
    else: 
        game_year = current_date.year #date corresponds to this year 
    return date(game_year, month_date.month, month_date.day)


'''
Functionality to order players into a dictionary based on their last name inital

Args:
    player_data(dict): dictionary containing unique players in current season

Returns:
    ordered_players(dict) : dictionary that orders players (i.e 'A': [{<player_data>}, {<player_data>}, ...])
'''
def order_players_by_last_name(player_data: dict): 
    logging.info("Ordering retrieved players by their last name's first initial")
    
    ordered_players = {
        "A": [],"B": [],"C": [],"D": [],"E": [],
        "F": [],"G": [],"H": [],"I": [],"J": [],
        "K": [],"L": [],"M": [],"N": [],"O": [],
        "P": [],"Q": [],"R": [],"S": [],"T": [],
        "U": [],"V": [],"W": [],"X": [],"Y": [],
        "Z": []
    }

    for player in player_data:
       last_name = player["player_name"].split()[1]
       inital = last_name[0].upper()
       ordered_players[inital].append(player)
    
    return ordered_players   



'''
Functionality to fetch all relevant player URLs needed to extract new player metrics 

Args:
    ordered_players(dict): players ordered by their last name inital, allowing us to construct 
                            player URLs in bulk rather than one by one
    year(int): season to fetch players for 
    
Returns:
    urls(list) : list of dictionary containing players URL and name
'''
def get_player_urls(ordered_players: dict, year: int): 
    base_url = "https://www.pro-football-reference.com%s/gamelog/%s"
    urls = [] 
    
    for inital, player_list in ordered_players.items(): 
        logging.info(f"Constructing player URLs for players with the inital \'{inital}\'")
        
        # fetch players list based on last name inital 
        player_list_url = "https://www.pro-football-reference.com/players/%s/" % (inital)
        soup = BeautifulSoup(fetch_page(player_list_url), "html.parser") # soup containing players hrefs
        
        # for each player in the corresponding inital, construct player URLs
        for player in player_list: 
            player_name = player['player_name']
            href = get_href(player_name, player['position'], year, soup)
            if(href == None):
                continue
            else:
                url = base_url % (href, year)
                urls.append({"player": player_name, "url": url}) # append each players URL to our list of URLs 
        
    return urls 


'''
Functionality to fetch a specific players href, which is needed to construct their URL

All credit for the following code in this function goes to the developer of the repository:
      - https://github.com/mjk2244/pro-football-reference-web-scraper

Args:
    player_name (str): players name to search for 
    year (int): year corresponding to the season we are searching for metrics for 
    soup (BeautifulSoup): soup pertaining to raw HTML containing players hrefs

Returns:
    href (str): href needed to construct URL 
'''
def get_href(player_name: str, position: str, year: int, soup: BeautifulSoup):
    players = soup.find('div', id='div_players').find_all('p')  # find players HTML element

    for player in players:
        # Split and parse the years from player text
        years = player.text.split(' ')
        years = years[-1].split('-')  # Use the last segment for the years

        try:
            start_year, end_year = int(years[0]), int(years[1])
        except ValueError:
            logging.warning(f"Error parsing years for player {player.text}")
            continue

        # Check if the player's name, position, and year match
        player_text = player.text
        if start_year <= year <= end_year and position in player_text and check_name_similarity(player_text, player_name) >= 90:
            a_tag = player.find('a')
            if a_tag and a_tag.get('href'):
                href = a_tag.get('href').replace('.htm', '')
                return href
            else:
                logging.warning(f"Missing href for player {player_name} ({position}) in year {year}")
                return None
    
    #TODO (FFM-40): Add Ability to Re-try Finding Players Name
    logging.warning(f"Cannot find a {position} named {player_name} from {year}")
    return None  

'''
Helper function to determine the similarity between two names

Args:
    player_name (str): players name to compare 
    player_text (str): text from PFR containing players name

Returns:
    similarity (float): similarity of the two passed in names
'''
def check_name_similarity(player_text: str, player_name: str):
    words = player_text.split()
    name = ' '.join(words[:2])
    return fuzz.partial_ratio(name, player_name)
