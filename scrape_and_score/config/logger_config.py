import logging
import os
from datetime import datetime

"""
Functionality to configure our logging format 

"""


def configure_logging():
    # create logging directory if not already created 
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', 'logs')
    os.makedirs(log_dir, exist_ok=True)


    time = datetime.now() 
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # set log file path 
    log_file = os.path.join(log_dir, f'app_{timestamp}.log')


    # configure logging 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # logs to stddout
        ]
    )
