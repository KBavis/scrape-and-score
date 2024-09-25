import logging 
from bs4 import BeautifulSoup
from .scraping_util import fetch_page



def scrape(url): 
   logging.info(f"Attempting to scrape the following URL: {url}")
     
   #Fetch Raw HTMl corresponding to Fantasy Pros site 
   html = fetch_page(url)
   
   #Parse with Beautiful Soup
   soup = BeautifulSoup(html, 'html.parser')
   
   # Fetch Players 
   return fetch_team_data(soup)

   

'''
Functionality to retrieve relevant fantasy football players and teams from current season

Args:
   teams (list) - all current NFL teams
   
Returns: 
   depth_chart (list of dictionaries) - each player corresponding to team 
'''

def fetch_team_data(soup): 
   logging.info(f"Fetching all relevant WR/RB/QB/TEs corresponding to each NFL team")
   
   team_data = []
   position_mapping = {
      "Quarterbacks": "QB",
      "Running Backs": "RB",
      "Wide Receivers": "WR", 
      "Tight Ends": "TE"
   }
   
   #Extract each HTML section corresponding to an NFL team 
   teams = soup.find_all("div", class_="team-list") 
   
   #Loop through each team 
   for team in teams:
      #Extract team name 
      team_name = team.find("input", class_="team-name")["value"]
      
      # Find each relevant list of players by position 
      positions = team.find_all("div", class_="position-list")
      
      #Loop through each relevant fantasy football position for current team
      for position in positions:
         #Extract position name 
         position_name = position.find("h4", class_="position-head").text.strip() 
         position_abrv = position_mapping.get(position_name[3:]) #map to abbreviated position and skip the 'ECR' substring at beginning
         
         #Find all players corresponding to position for current team 
         players = position.find_all("a", class_="fp-player-link")
         
         #Loop through each player and extract player name 
         for player in players:
            player_name = player['fp-player-name']
            
         #Append Extract Data to Output 
         team_data.append({
            'team': team_name, 
            'position': position_abrv,
            'player_name': player_name
         })   
   
   return team_data 

            