from config import configure_logging
from scraping import scrape_fantasy_pros
from scraping import pfr_scraper as pfr
from db import connection, is_player_empty, is_team_empty, insert_teams, fetch_all_teams, insert_players, fetch_all_players
from config import load_configs, get_config
from datetime import datetime
from service import player_game_logs_service, team_game_logs_service
from data import preprocess
from models.lin_reg import LinReg
import logging
import sys


'''
   Main entry point of our application that will initate web scraping, cleaning of data, 
   persisting of data, generation of model, generation of predictions, and generating output
'''
def main():
      
   # parse command line args
   args = ['--help', '--recent', '--new', '--test', '--single-player', '--all-players']
   cl_args = sys.argv[1:]
   n = len(cl_args)
   
   if n == 0:
      print(f"Error: Please pass in one of the following command line args: {args}")
      exit(-1)
   else:
      if args[0] in cl_args: 
         msg = '''
            Utilize one of the followinging command line arguments to invoke this application 
            
            --help: indicates possible command line args and their purpose 
            --recent: scrape most recent game logs for teams & players 
            --new: scrape for all available NFL players & teams, along with all their corresponding game logs 
            --test: skip scraping altogether and test our regression against via our test data 
            --single-player: prompt user to enter a single players name & matchup and indicate fantasy point prediction 
            --all-players: automatically fetch all upcoming matchups for fantasy relevant players and generate predictions
         '''
         print(msg)
         exit(1)
   
   try:
      configure_logging()
      
      load_configs() 
      connection.init()
      start_time = datetime.now()
   
      template_url = get_config('website.fantasy-pros.urls.depth-chart')
      teams = [team['name'] for team in get_config('nfl.teams')] # extract teams from configs
      year = get_config('nfl.current-year')
      team_names_and_ids = []  
      # check if teams are persisted; if not, persist relevant teams
      if is_team_empty():
         logging.info('No teams persisted; persisting all configured NFL teams to our database')
         team_names_and_ids = insert_teams(teams)
      else: 
         logging.info('All teams persisted; skipping insertion and fetching teams from database')
         team_names_and_ids = fetch_all_teams()
   
   
      # check if players are persisted; if not, persist relevant players
      depth_charts = []
      if is_player_empty():
            
         depth_charts = scrape_fantasy_pros(template_url, team_names_and_ids)
         logging.info(f"Successfully fetched {len(depth_charts)} unique fantasy relevant players and their corresponding teams")
         
         # insert players into db 
         logging.info('Inserting fantasy relevant players into db')
         insert_players(depth_charts)
      else: 
         # TODO: FFM-64 - Check for depth chart changes to determine if need to make updates to persisted data
         logging.info('All players persisted; skipping insertion')
   
      # ensure that we fetch player records (need player ID)
      depth_charts = fetch_all_players() 
      
      # fetch relevant team and player metrics 
      if player_game_logs_service.is_player_game_logs_empty(): # scrape & persist all game logs if none persisted
         team_metrics, player_metrics = pfr.scrape_all(depth_charts, teams)
         logging.info(f"Successfully retrieved metrics for {len(team_metrics)} teams and {len(player_metrics)} players")
         
         # insert into player_game_log & team_game_log
         player_game_logs_service.insert_multiple_players_game_logs(player_metrics, depth_charts)
         team_game_logs_service.insert_multiple_teams_game_logs(team_metrics, team_names_and_ids)
         
         #calculate fantasy points for each week
         player_game_logs_service.calculate_fantasy_points(False, year) 
      else :
         logging.info('All previous games fetched; fetching metrics for most recent week')
         team_metrics, player_metrics = pfr.scrape_recent()
         logging.info(f"Successfully retrieved most recent game log metrics for {len(team_metrics)} teams and {len(player_metrics)} players")
         
         # insert into player_game_log & team_game_log
         player_game_logs_service.insert_multiple_players_game_logs(player_metrics, depth_charts) 
         team_game_logs_service.insert_multiple_teams_game_logs(team_metrics, team_names_and_ids) 
         logging.info('Successfully persisted most recent player & game log metrics')
         
         # calculate fantasy points for recent week 
         player_game_logs_service.calculate_fantasy_points(True, year)
         

      team_game_logs_service.calculate_all_teams_rankings(year)  
      qb_pre_processed_data, rb_pre_processed_data, wr_pre_processed_data, te_pre_processed_data = preprocess.pre_process_data()
      
      linear_regressions = LinReg(qb_pre_processed_data, rb_pre_processed_data, wr_pre_processed_data, te_pre_processed_data)
      linear_regressions.create_regressions()
      linear_regressions.test_regressions()
      

      logging.info(f'Application took {(datetime.now() - start_time).seconds} seconds to complete')
   
   except Exception as e:
      logging.error(f'An exception occured while executing the main script: {e}')
   finally:
      connection.close_connection()
   
# entry point 
if __name__ == "__main__":
    main()