import logging
from config import configure_logging
from scraping import scrape_fantasy_pros
from scraping import fetch_metrics


'''
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
'''
def main():
   #Set Up Global Logger Configurations
   configure_logging()
   
   #Fetch relevant NFL teams & players
   team_data = scrape_fantasy_pros("https://www.fantasypros.com/nfl/depth-charts.php")
   
   #Fetch relevant team and player metrics 
   fetch_metrics(team_data)

# Entry point for the script
if __name__ == "__main__":
    main()