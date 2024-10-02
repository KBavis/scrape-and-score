from config import configure_logging
from scraping import scrape_fantasy_pros
from scraping import scrape_pro_football_reference
from config import load_configs


'''
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
'''
def main():
   #Set Up Global Logger Configurations
   configure_logging()
   
   #Load Configurations from YAML file
   config = load_configs("../resources/application.yaml") 
   
   #Fetch relevant NFL teams & players
   teams_and_players = scrape_fantasy_pros(config['website']['fantasy-pros']['urls']['depth-chart'])
   
   #Fetch relevant team and player metrics 
   team_metrics, player_metrics = scrape_pro_football_reference(teams_and_players, config['nfl']['current-year'], config)

# Entry point for the script
if __name__ == "__main__":
    main()