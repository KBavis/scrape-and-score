import argparse

"""
Functionality to parse our command line arguments to determine the type of program execution we will be running 

Args:
    None 

Returns:
    args (namespace): obj containing arguments
"""


def parse():
    parser = argparse.ArgumentParser(prog="Scrape and Score")

    # define groups for validation
    model_group = parser.add_mutually_exclusive_group()
    data_collection_group = parser.add_mutually_exclusive_group()

    # establish independent args 
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Re-train our neural network model."
    )

    # data collction args
    data_collection_group.add_argument(
        "--historical",
        nargs=2, 
        metavar=("START_YEAR", "END_YEAR"),
        type=int,
        help="Scrape & persist historical data for players & teams from START_YEAR to END_YEAR."
    )
    data_collection_group.add_argument(
        "--upcoming",
        nargs=2,
        metavar=("WEEK", "SEASON"),
        type=int,
        help="Scrape & persist upcoming data for players & teams for the specified week & season"
    )
    data_collection_group.add_argument(
        "--update",
        nargs=2,
        metavar=("WEEK", "SEASON"),
        type=int,
        help="Scrape & persist finalized outcomes for teams/players for the specified week & season"
    )

    # model selection args
    model_group.add_argument(
        "--nn",
        action="store_true",
        default=False,
        help="Utilize neural network capabilities for our predictions"
    )
    model_group.add_argument(
       "--lin_reg",
       action="store_true",
       default=False,
       help="Utilize linear regression model capabilities for our predictions" 
    )

    # parse args
    args = parser.parse_args()

    return args
