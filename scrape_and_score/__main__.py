from config import configure_logging
from scraping import scrape_fantasy_pros
from scraping import pfr_scraper as pfr
from db import connection, is_player_empty, is_team_empty, insert_teams, fetch_all_teams, insert_players, fetch_all_players
from config import load_configs, get_config
import logging


'''
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
'''
def main():
   
   try:
      configure_logging()
      load_configs() 
      connection.init()
   
      template_url = get_config('website.fantasy-pros.urls.depth-chart')
      teams = [team['name'] for team in get_config('nfl.teams')] # extract teams from configs
      team_names_and_ids = []
   
      # check if teams are persisted; if not, persist relevant teams
      if is_team_empty():
         logging.info('No teams persisted; persisting all configured NFL teams to our database')
         team_names_and_ids = insert_teams(teams)
      else: 
         logging.info('All teams persisted; skipping insertion')
   
   
      # check if players are persisted; if not, persist relevant players
      depth_charts = []
      if is_player_empty():
         # fetch relevant team names & ids if needed
         if len(team_names_and_ids) == 0:
            logging.info('Fetching all relevant NFL teams')
            team_names_and_ids = fetch_all_teams()
            
         depth_charts = scrape_fantasy_pros(template_url, team_names_and_ids)
         logging.info(f"Successfully fetched {len(depth_charts)} unique fantasy relevant players and their corresponding teams")
         
         # insert players into db 
         logging.info('Inserting fantasy relevant players into db')
         insert_players(depth_charts)
      else: 
         # TODO: FFM-64 - Check for depth chart changes to determine if need to make updates to persisted data
         logging.info('All players persisted; skipping insertion')
   
   
      if len(depth_charts) == 0: 
         depth_charts = fetch_all_players() 
         
      # fetch relevant team and player metrics 
      team_metrics, player_metrics = pfr.scrape(depth_charts, teams)
      logging.info(f"Successfully retrieved metrics for {len(team_metrics)} teams and {len(player_metrics)} players")
   
   except Exception as e:
      logging.error('An exception occured while executing the main script', e)
   finally:
      connection.close_connection()
   
# entry point 
if __name__ == "__main__":
    main()