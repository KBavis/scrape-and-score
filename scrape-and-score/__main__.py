from config import configure_logging
from scraping import scrape_fantasy_pros
from scraping import pfr_scraper as pfr
from config import load_configs
import logging


'''
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
'''
def main():
   #Set Up Global Logger Configurations
   configure_logging()
   
   #Load Configurations from YAML file
   config = load_configs() 
   
   #Fetch relevant NFL teams & players
   teams_and_players = scrape_fantasy_pros(config['website']['fantasy-pros']['urls']['depth-chart'])
   logging.info(f"Successfully fetched {len(teams_and_players)} unique fantasy relevant players and their corresponding teams")
   
   #Fetch relevant team and player metrics 
   team_metrics, player_metrics = pfr.scrape(teams_and_players, config['nfl']['current-year'])
   logging.info(f"Successfully retrieved metrics for {len(team_metrics)} teams and {len(player_metrics)} players")
   
# Entry point for the script
if __name__ == "__main__":
    main()