from config import configure_logging
from db import (
    connection,
    insert_teams,
    fetch_all_teams,
)
from config import load_configs, get_config
from datetime import datetime
from util import args
import logging
from workflows import workflows




def main():
    """
    Main entry point of application that will invoke the necessary workflows based on the command line arguments passed
    """

    cl_args = args.parse()

    try:
        # init configs, logging, and db connection
        configure_logging()
        load_configs()
        connection.init()
        start_time = datetime.now()



        # account for teams potentially not being persisted yet
        teams = fetch_all_teams() 
        if not teams:
            teams = [
                team["name"] for team in get_config("nfl.teams")
            ]             
            insert_teams(teams)


        # data collection workflows
        if cl_args.upcoming: 
            week, season = cl_args.upcoming
            logging.info(f"---------------'Upcoming' Wofklow Invoked: Scraping & persisting upcoming player & team data for [Week: {week}, Season: {season}]---------------")
            workflows.upcoming(week, season)
        elif cl_args.historical:
            start_year, end_year = cl_args.historical
            logging.info(f"---------------'Historical' Wofklow Invoked: Scraping & persisting player & team data for [Start Year: {start_year}, End Year: {end_year}]---------------")
            workflows.historical(start_year, end_year)
        elif cl_args.results: 
            week, season = cl_args.results
            logging.info(f"---------------'Results' Wofklow Invoked: Updating player & team data based on game outcomes for [Week: {week}, Season: {season}]---------------")
            workflows.historical(start_year, end_year)


        
        # model workflows 
        if cl_args.lin_reg:
            logging.info(f"---------------'Linear Regression' Wofklow Invoked: Generating Linear Regression for predicting players fantasy points---------------")
            workflows.linear_regression()
        elif cl_args.nn:
            logging.info(f"---------------'Neural Network' Wofklow Invoked: Generating Neural Network for predicitng players fantasy points---------------")
            workflows.neural_network(cl_args.train, start_time)


    except Exception as e:
        logging.error(f"An exception occured while executing the main script: {e}")
    finally:
        connection.close_connection()
        logging.info(
            f"Application took {(datetime.now() - start_time).seconds} seconds to complete"
        )


# entry point
if __name__ == "__main__":
    main()
