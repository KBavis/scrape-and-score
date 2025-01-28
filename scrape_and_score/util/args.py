import argparse

'''
Functionality to parse our command line arguments to determine the type of program execution we will be running 

Args:
    None 

Returns:
    args (namespace): obj containing arguments
'''
def parse():
    parser = argparse.ArgumentParser(prog='Scrape and Score')
    
    # define groups for validation
    scraping_group = parser.add_mutually_exclusive_group() 
    input_group = parser.add_mutually_exclusive_group()
    betting_group = parser.add_mutually_exclusive_group()

    # add arguments with default=False
    parser.add_argument("--test", action="store_true", default=False, help="Skip scraping and perform testing of regression models.")
    scraping_group.add_argument("--recent", action="store_true", default=False, help="Scrape most recent game logs for teams and players.")
    scraping_group.add_argument("--new", action="store_true", default=False, help="Scrape all available NFL players, teams, and game logs.")
    input_group.add_argument("--single_player", action="store_true", default=False, help="Prompt for a single player's name and matchup for fantasy point prediction.")
    input_group.add_argument("--all_players", action="store_true", default=False, help="Fetch all upcoming matchups for relevant players and generate predictions.")
    betting_group.add_argument("--upcoming", action="store_true", default=False, help="Scrape and persist upcoming NFL games betting odds.")
    betting_group.add_argument("--update_odds", action="store_true", default=False, help="Update previously persisted betting odds with game outcomes.")
    betting_group.add_argument("--historical", action="store_true", default=False, help="Scrape and persist all previous betting odds for a particular season.")

    # parse args 
    args = parser.parse_args()

    # Provide feedback on which flags are set
    if args.recent:
        print("--recent flag passed: Scraping and persisting game logs for the most recent week.")
    if args.new:
        print("--new flag passed: Scraping and persisting all game logs for the season.")
    if args.single_player:
        print("--single_player flag passed: Predicting for a single player and matchup.")
    if args.all_players:
        print("--all_players flag passed: Predicting for all relevant players and matchups.")
    if args.test:
        print("--test flag passed: Performing regression model testing.")
    if args.upcoming:
        print("--upcoming flag passed: Scraping and persisting upcoming NFL games betting odds.")
    if args.update_odds:
        print("--update_odds flag passed: Updating all previously persisted betting odds with game outcomes.")
    if args.historical:
        print("--historical flag passed: Scraping and persisting all previous betting odds for a given season.")

    return args
