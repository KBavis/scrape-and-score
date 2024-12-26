import logging 
from bs4 import BeautifulSoup
from .util import fetch_page


'''
Functionality to fetch all the relevant fantasy football players and their respective teams 

Args:
   base_url (str): tempalte URL for a teams depth chart 
   teams (list): list of all NFL teams & corresponding IDs 

Returns: 
   team_and_player_data (list): list containing all team and player data 
'''
def scrape(base_url: str, teams: list): 
   
   logging.info("Beginning to scrape Fantasy Pros for fantasy relevant NFL Players & Teams")
   team_and_player_data = []
   
   for team in teams: 
      # construct url 
      url = construct_url(team['name'], base_url)

      # fetch raw HTML 
      html = fetch_page(url)
      
      # parse with Beautiful Soup
      soup = BeautifulSoup(html, 'html.parser')
      
      # fetch depth chart
      team_depth_chart = get_depth_chart(soup, team['name'], team['team_id'])
      
      # add teams player data to our return value
      team_and_player_data.extend(team_depth_chart)

   
   return team_and_player_data


'''
Functionality to construct the proper URL to fetch teams depth cahrt 

Args:
   team (str): NFL team name 
   base_url (str): template URL to update 

Returns:
   formatted_url (str): url for a particular teams depth chart 
'''
def construct_url(team: str, base_url: str):
   formatted_team_name = team.lower().replace(" ", "-")
   return base_url.replace("{TEAM}", formatted_team_name)



'''
Functionality to parse HTML and extract relevant players for a particular team 

Args:
   soup (BeautifulSoup): parsed HTML with team data 
   team (str): team name 
   team_id (int): ID of team
'''
def get_depth_chart(soup: BeautifulSoup, team: str, team_id: int):
   logging.info(f"\nFetching depth chart for the following team: {team}")
   
   tables = soup.find_all("table") 
   players_data = []
   
   for table in tables:
      rows = table.find_all("tr")  
    
      for row in rows:
         cells = row.find_all("td")
        
         if len(cells) >= 2:  # ensure cell exists for name & position
               position = cells[0].get_text(strip=True) 
               name = cells[1].get_text(strip=True)

               # ensure cells contains valid data
               if position == "" or name == "":
                  continue
               
               # add player 
               players_data.append({"player_name": name, "position": position[:2], "team_id": team_id})
   
   logging.info(f"Relevant players for the NFL Team '{team}: {players_data}\n")
   return players_data

            