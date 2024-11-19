import logging;

'''
Functionality to configure our logging format 

'''
def configure_logging():
    logging.basicConfig(level=logging.INFO,  
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S') 
