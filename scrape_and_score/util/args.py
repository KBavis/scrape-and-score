import argparse


def parse():
    """
    Functionality to parse our command line arguments to determine the type of program execution we will be running 

    Returns:
        args (namespace): obj containing arguments
    """

    parser = argparse.ArgumentParser(prog="Scrape and Score")

    # define groups for validation
    model_group = parser.add_mutually_exclusive_group()
    data_collection_group = parser.add_mutually_exclusive_group()

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
        "--results",
        nargs=2,
        metavar=("WEEK", "SEASON"),
        type=int,
        help="Scrape & persist data following finalized outcomes of games for teams/players for the specified week & season"
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


    # establish independent args 
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Re-train our neural network model."
    )
    parser.add_argument(
        "--predict",
        metavar=("WEEK", "SEASON"), 
        nargs=2,
        type=int,
        help="Generate top 40 fantasy point predictions for each position for a specific week & season"
    )


    # parse args
    args = parser.parse_args()

    return args
