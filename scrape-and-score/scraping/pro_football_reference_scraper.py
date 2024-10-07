import logging
from datetime import date, datetime
import pandas as pd
from bs4 import BeautifulSoup
from pro_football_reference_web_scraper import player_game_log as p
from constants import team_hrefs, months, locations, cities, valid_positions
from scraping import fetch_page
from haversine import haversine, Unit
from .player_game_log import get_player_game_log


'''
Functionality to fetch metrics for all relevant players and teams for 
the specified NFL season via pro-football-reference
   
Args:
    team_and_player_data (list[dict]): every relevant fantasy NFL player corresponding to specified NFL season
    year (int): specified NFL season
    config (obj): object storing all YAML configurations
   
Returns:
    tuple(list[dict(str, pandas.DataFrame)]): A tuple (one for players and one for teams) containing two lists of a dictionary, 
    with each dictionary containing a string and pandas.DataFrame
'''
def fetch_metrics(team_and_player_data, year, config): 
    logging.info(f"Attempting to fetch the relevant player and team metrics for the year {year}")
    
    #Extract Unique Teams 
    unique_teams = {team['team'] for team in team_and_player_data}
    
    #Extract Relevant Metrics for Each Team
    team_metrics = []  
    for team_name in unique_teams:
        logging.info(f"Extracting team metrics for the NFL Team \'{team_name}\'")
        # get raw html
        raw_html = get_team_metrics_html(team_name, year, config['website']['pro-football-reference']['urls']['team-metrics'])
        
        
        if(raw_html == None):
            logging.error(f'An error occured while fetching raw HTML for the team \'{team_name}\'')
            raise Exception(f"Unable to extract raw HTML for the NFL Team \'{team_name}\'")
        
        # create BeautifulSoup instance 
        soup = BeautifulSoup(raw_html, "html.parser")
        
        # fetch relevant team metrics 
        team_data = collect_team_data(soup, year, team_name)
        
        # add to list 
        team_metrics.append({"team_name": team_name, "team_metrics": team_data})
    
    
    #Extract Relevant Metrics for Each Player
    player_metrics = []
    for player in team_and_player_data:
        
        # fetch player metrics 
        player_data = collect_player_data(player['player_name'], player['position'], year)
        
        # note that the player should be removed if unable to fetch metrics
        if(player_data == None):
            continue
        
        # add to list
        player_metrics.append({"player_name": player["player_name"], "player_metrics": player_data})

    
    return team_metrics,player_metrics    
        



'''
Functionality to fetch the relevant metrics to a specific player


   
Args:
    player (str): NFL player's full name
   
Returns:
    pandas.DataFrame: A pandas DataFrame with relevant metrics corresponding to the specific player
'''
def collect_player_data(name: str, position: str, season: int):
    try: 
        logging.info(f"Attempting to collect player metrics for player \'{name}\'")
        return get_player_game_log(name, position, season)
    except Exception as e:
        logging.error(f"An error occured while fetching metrics for player \'{name}\': {str(e)}") 
        return None
           



'''
Functionality to generate URL and fetch raw HTML for NFL Team

Args: 
    team_name (str) - NFL team full name
    year (int) - year to fetch raw HTML for
    url (str) - template URL to fetch HTML from
      
Returns:
    str - raw HTML from web page 
''' 
def get_team_metrics_html(team_name, year, url):
   url = url.replace("{TEAM_ACRONYM}", team_hrefs[team_name]).replace("{CURRENT_YEAR}", str(year))
   return fetch_page(url)


'''
Functionality to fetch relevant metrics corresponding to a specific NFL team

All credit for the following code in this function goes to the developer https://github.com/mjk2244. 
The code within their created library (https://github.com/mjk2244/pro-football-reference-web-scraper) 
contains a bug, which I have fixed and added an issue for their library. 
   
Args: 
    team (str) - NFL team full name
      
Returns:
    pandas.DataFrame: A pandas DataFrame with relevant metrics corresponding to the specific player     
''' 
def collect_team_data(soup: BeautifulSoup, season: int, team: str):
    
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
                rest_days = date(season + 1, months[date2[0]], int(date2[1])) - date(
                    season + 1, months[date1[0]], int(date1[1])
                )
            elif date2[0] == 'January':
                rest_days = date(season + 1, months[date2[0]], int(date2[1])) - date(
                    season, months[date1[0]], int(date1[1])
                )
            else:
                rest_days = date(season + 1, months[date2[0]], int(date2[1])) - date(
                    season + 1, months[date1[0]], int(date1[1])
                )
        else:
            rest_days = date(2022, 7, 11) - date(2022, 7, 1)  # setting first game as 10 rest days

        opp = games[i].find('td', {'data-stat': 'opp'}).text

        if games[i].find('td', {'data-stat': 'game_location'}).text == '@':
            home_team = False
            distance_travelled = calculate_distance(locations[cities[team]], locations[cities[opp]])
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
Functionality to calculate the distance between two cities 

All credit for the following code in this function goes to the developer https://github.com/mjk2244. 
The code within their created library (https://github.com/mjk2244/pro-football-reference-web-scraper) 
contains a bug, which I have fixed and added an issue for their library. 

   
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