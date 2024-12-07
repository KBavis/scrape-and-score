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
   teams (list): list of unique NFL team names

Returns:
   data (tuple(list[pd.DataFrame], list[pd.DataFrame])) 
      - metrics for both players and teams
'''
def scrape(team_and_player_data: list, teams: list):
   # TODO (FFM-31): Create logic to determine if new player/team data avaialable. If no new team data available, skip fetching metrics and utilize persisted metrics. If no new player data available, skip fetching metrics for player.
   
   # fetch configs 
   team_template_url = props.get_config('website.pro-football-reference.urls.team-metrics')
   year = props.get_config('nfl.current-year')

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
   player_metrics = []
    
   # order players by last name inital 
   ordered_players = order_players_by_last_name(team_and_player_data)
       
   # construct each players metrics link 
   player_urls = get_player_urls(ordered_players, year)
   
   # for each player url, fetch relevant metrics 
   for player_url in player_urls:
       url = player_url['url']
       player_name = player_url['player']
       position = player_url['position']
       
       raw_html = fetch_page(url)
       if raw_html == None: #ensure we retrieve response prior to parsing
           continue
       
       soup = BeautifulSoup(raw_html, "html.parser")
       
       #TODO (FFM-42): Gather Additional Data other than Game Logs  
       logging.info(f"Fetching metrics for {position} \'{player_name}\'")
       player_metrics.append({"player": player_name,"position": position, "player_metrics": get_game_log(soup, position)})

   
   return player_metrics    


'''
Functionality to fetch the metrics for each NFL team 

Args:
   teams (list) - list of team names to fetch metrics for 
   url_template (str) - template URL used to construct specific teams URL
   year (int) - year to fetch metrics for 

Returns:
    team_metrics (list) - list of df's containing team metrics
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
         raise Exception(f"Unable to collect team data for the NFL Team \'{team}\'")
      
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
    
    # configure data frame
    data = {
        'week': [],
        'day': [],
        'rest_days': [],
        'home_team': [],
        'distance_traveled': [],
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

    #load game data 
    games = soup.find_all('tbody')[1].find_all('tr')
    
    remove_uneeded_games(games) 
      
    # gathering data for each game
    for i in range(len(games)):
        week = int(games[i].find('th', {'data-stat': 'week_num'}).text)
        day = games[i].find('td', {'data-stat': 'game_day_of_week'}).text
        
        rest_days = calculate_rest_days(games, i, year)
        
        opp = games[i].find('td', {'data-stat': 'opp'}).text

        if games[i].find('td', {'data-stat': 'game_location'}).text == '@':
            home_team = False
            distance_travelled = calculate_distance(LOCATIONS[CITIES[team]], LOCATIONS[CITIES[opp]])
        else:
            home_team = True
            distance_travelled = 0

        result = games[i].find('td', {'data-stat': 'game_outcome'}).text
        points_for = extract_int(games[i], 'pts_off')
        points_allowed = extract_int(games[i], 'pts_def')

        
        tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds = calculate_yardage_totals(games, i)

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
Helper function to calculate the yardage totals for a particular game of a team 

Args: 
    games (BeautifulSoup): parsed HTML containing game data 
    index (int): index pertaining to current game 

Returns:
    tot_yds,pass_yds,rush_yds,opp_tot_yds,opp_pass_yds,opp_rush_yds (tuple): yardage totals of particular game 
'''
def calculate_yardage_totals(games: BeautifulSoup, index: int): 
    tot_yds = extract_int(games[index], 'yards_off')
    pass_yds = extract_int(games[index], 'pass_yds_off')
    rush_yds = extract_int(games[index], 'rush_yds_off')
    opp_tot_yds = extract_int(games[index], 'yards_def')
    opp_pass_yds = extract_int(games[index], 'pass_yds_def')
    opp_rush_yds = extract_int(games[index], 'rush_yds_def')

    return tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds

'''
Helper function to determine a teams total rest days 

Args:
    games (BeautifulSoup): parsed HTML containing game data 
    index (int): index pertaining to current game 
    year (int): year we are calculating metrics for 

Returns:
    rest_days (int): total number of rest days since previous game 

'''
def calculate_rest_days(games: BeautifulSoup, index: int, year: int):
        if index == 0:
            return 10 # set rest days to be 10 if first game of year 
        
        # fetch previous game and current game date
        previous_game_date = games[index - 1].find('td', {'data-stat': 'game_date'}).text.split(' ')
        current_game_date= games[index].find('td', {'data-stat': 'game_date'}).text.split(' ')
        
        # account for new year 
        if current_game_date[0] == 'January' and previous_game_date[0] != "January":
            return date(year + 1, MONTHS[current_game_date[0]], int(current_game_date[1])) - date(year, MONTHS[previous_game_date[0]], int(previous_game_date[1]))
        # both games in new year
        elif current_game_date[0] == 'January' and previous_game_date[0] == "January":
            return date(year + 1, MONTHS[current_game_date[0]], int(current_game_date[1])) - date(year + 1, MONTHS[previous_game_date[0]], int(previous_game_date[1]))
        # both games not in new year
        else:
            return date(year, MONTHS[current_game_date[0]], int(current_game_date[1])) - date(year, MONTHS[previous_game_date[0]], int(previous_game_date[1]))

'''
Helper function to remove all canceled/playoff games, bye weeks, 
and games yet to be played so that they aren't accounted for 

Args: 
    games (BeautifulSoup): parsed HTML containing game data 
'''
def remove_uneeded_games(games: BeautifulSoup):
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

Note: This application should be run on a scheduled basis throughout the football year, 
starting in August and concluding in Feburary. Therefore, the game date year should account
for this logic 

Args:
    game (BeautifulSoup): BeautifulSoup object containing relevant game data 
    current_date (date) : the current date the application is being run 

Returns:
    game_date (date) : The date corresponding to the game day    
'''
def get_game_date(game: BeautifulSoup, current_date: date): 
    game_date_str = game.find('td', {'data-stat': 'game_date'}).text
    month_date = datetime.strptime(game_date_str, "%B %d").date() #convert to date time object
    
    # if game date month is janurary or feburary, we must adjust logic 
    if month_date.month == 1 or month_date.month == 2: 
        if current_date.month == 1 or current_date.month == 2: 
            game_year = current_date.year # game year should be current year if application is running in january or febutary 
        else:
            game_year = current_date.year + 1 # game yar should be next year if application is running in march through december
    else: 
        game_year = current_date.year #date corresponds to this year if game date isn't jan or feb
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
            player_position = player['position']
            
            href = get_href(player_name, player_position, year, soup) # extract href from parsed HTML  
            if(href == None):
                continue
            else:
                url = base_url % (href, year)
                urls.append({"player": player_name,"position": player_position, "url": url}) # append each players URL to our list of URLs 
        
    return urls 


'''
Functionality to fetch a specific players href, which is needed to construct their URL

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



'''
Functionality to get the game log for a player. This function 
will retrieve relevant player metrics from their game logs based 
on their position.

Args:
    soup (BeautifulSoup): parsed HTML containing relevant player metrics 
    position (str): the players corresponding position

Returns:
    data (pd.DataFrame): data frmae containing player game logs 
'''
def get_game_log(soup: BeautifulSoup, position: str):
    # data to retrieve for each player, regardless of position
    data = {
        'date': [],
        'week': [],
        'team': [],
        'game_location': [],
        'opp': [],
        'result': [],
        'team_pts': [],
        'opp_pts': [],
    }
    data.update(get_additional_metrics(position)) # update data with additonal metrics 
    
    table_rows = soup.find('tbody').find_all('tr')
    
    # ignore inactive/DNP games 
    ignore_statuses = ['Inactive', 'Did Not Play', 'Injured Reserve']
    filtered_table_rows = [] 
    for tr in table_rows:
        elements = tr.find_all('td')
        status = elements[-1].text
        
        if status not in ignore_statuses:
            filtered_table_rows.append(tr)
    
    # add game log data to dictionary  
    for tr in filtered_table_rows:
        # add common game log data 
        add_common_game_log_metrics(data, tr)
        
        # add position specific data
        if position == 'QB':
            add_qb_specific_game_log_metrics(data, tr)
        elif position == 'RB':
            add_rb_specific_game_log_metrics(data,tr)
        else:
            add_wr_specific_game_log_metrics(data, tr)        
    
    return pd.DataFrame(data=data) 

'''
Functionality to retireve game log metrics for a QB

Args:
    tr (BeautifulSoup): parsed HTML tr containing player metrics 
    data (dict): dictionary containing players metrics 

Returns:
    None
''' 
def add_qb_specific_game_log_metrics(data: dict, tr: BeautifulSoup):
    data['cmp'].append(extract_int(tr, 'pass_cmp'))
    data['att'].append(extract_int(tr, 'pass_att'))
    data['pass_yds'].append(extract_int(tr, 'pass_yds'))
    data['pass_td'].append(extract_int(tr, 'pass_td'))
    data['int'].append(extract_int(tr, 'pass_int'))
    data['rating'].append(extract_float(tr, 'pass_rating'))
    data['sacked'].append(extract_int(tr, 'pass_sacked'))
    data['rush_att'].append(extract_int(tr, 'rush_att'))
    data['rush_yds'].append(extract_int(tr, 'rush_yds'))
    data['rush_td'].append(extract_int(tr, 'rush_td'))

'''
Functionality to retireve game log metrics for a RB

TODO (FFM-83): Account for RB Snap Percentage Metrics

Args:
    tr (BeautifulSoup): parsed HTML tr containing player metrics 
    data (dict): dictionary containing players metrics 

Returns:
    None
'''    
def add_rb_specific_game_log_metrics(data: dict, tr: BeautifulSoup): 

    # Add rushing and receiving stats with missing value handling
    data['rush_att'].append(extract_int(tr, 'rush_att'))
    data['rush_yds'].append(extract_int(tr, 'rush_yds'))
    data['rush_td'].append(extract_int(tr, 'rush_td'))
    data['tgt'].append(extract_int(tr, 'targets'))
    data['rec'].append(extract_int(tr, 'rec'))
    data['rec_yds'].append(extract_int(tr, 'rec_yds'))
    data['rec_td'].append(extract_int(tr, 'rec_td'))


'''
Functionality to retrieve game log metrics for a WR

Args:
    tr (BeautifulSoup): parsed HTML tr containing player metrics 
    data (dict): dictionary containing players metrics 

Returns:
    None
'''
def add_wr_specific_game_log_metrics(data: dict, tr: BeautifulSoup):
    data['tgt'].append(extract_int(tr, 'targets'))
    data['rec'].append(extract_int(tr, 'rec'))
    data['rec_yds'].append(extract_int(tr, 'rec_yds'))
    data['rec_td'].append(extract_int(tr, 'rec_td'))

    # Handle snap percentage
    snap_pct_td = tr.find('td', {'data-stat': 'off_pct'})
    if snap_pct_td and snap_pct_td.text:  # Check for valid data
        snap_pct = snap_pct_td.text[:-1]  # Remove '%' symbol
        data['snap_pct'].append(float(snap_pct) / 100)  # Convert to float percentage
    else:
        data['snap_pct'].append(0.0)  # Append 0 if snap percentage is not available
        
    
'''
Functionality to retireve common game log metrics for a given player 

Args:
    tr (BeautifulSoup): parsed HTML tr containing player metrics 
    data (dict): dictionary containing players metrics 

Returns:
    None
'''
def add_common_game_log_metrics(data: dict, tr: BeautifulSoup):
    data['date'].append(tr.find('td', {'data-stat': 'game_date'}).text)
    data['week'].append(int(tr.find('td', {'data-stat': 'week_num'}).text))
    data['team'].append(tr.find('td', {'data-stat': 'team'}).text)
    data['game_location'].append(tr.find('td', {'data-stat': 'game_location'}).text)
    data['opp'].append(tr.find('td', {'data-stat': 'opp'}).text)
    
    # For result, team_pts, and opp_pts, split the text properly
    game_result_text = tr.find('td', {'data-stat': 'game_result'}).text.split(' ')
    data['result'].append(game_result_text[0])
    data['team_pts'].append(int(game_result_text[1].split('-')[0]))
    data['opp_pts'].append(int(game_result_text[1].split('-')[1])) 


'''
Helper function to retrieve additional metric fields needed for a player based on position

Args:
    position (str): the players corresponding position

Returns:
    additonal_metrics (dict): additional metrics to account for based on player posiiton
'''
def get_additional_metrics(position):
    if position == 'QB':
        additional_fields = {
            'cmp': [],
            'att': [],
            'pass_yds': [],
            'pass_td': [],
            'int': [],
            'rating': [],
            'sacked': [],
            'rush_att': [],
            'rush_yds': [],
            'rush_td': [],
        }
    elif position == 'RB':
        additional_fields = {
            'rush_att': [],
            'rush_yds': [],
            'rush_td': [],
            'tgt': [],
            'rec': [],
            'rec_yds': [],
            'rec_td': [],
        }
    elif position == 'WR' or position == 'TE': 
        additional_fields = {
            'tgt': [],
            'rec': [],
            'rec_yds': [],
            'rec_td': [],
            'snap_pct': [],
        }
    else:
        logging.error(f"An unexpected position was passed to get_additional_metrics: {position}")
        raise Exception(f"The position '{position}' is not a valid position to fetch metrics for.")   
    return additional_fields    


'''
Helper function to extract int from a speciifc player metric 

Args:
    tr (BeautifulSoup): table row containing relevant metrics 

Returns:
    metric (int): derived metric converted to a int
'''
def extract_int(tr, stat):
    text = tr.find('td', {'data-stat': stat})
    
    if text == None:
        return 0 # return 0 if no value 
    elif text.text == '':
        return 0
    else:
        return int(text.text)

'''
Helper function to extract a float from a speciifc player metric 

Args:
    tr (BeautifulSoup): table row containing relevant metrics 

Returns:
    metric (float): derived metric converted to a float 
'''
def extract_float(tr, stat):
    text = tr.find('td', {'data-stat': stat})
    
    if text == None:
        return 0.0
    elif text.text == '':
        return 0.0
    else:
        return float(text.text) 