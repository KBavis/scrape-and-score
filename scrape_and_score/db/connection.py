from psycopg2 import connect
from dotenv import load_dotenv
import os
import logging


'''
Intialize a global DB connection 
'''
def init():
   global _connection 
   load_dotenv()
   
   try:
      _connection = connect(database=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), \
                        password=os.getenv("db.password"), host=os.getenv("DB_HOST"), \
                        port=os.getenv("DB_PORT"))
      logging.info('Successfully established database connection')
   except Exception as e:
      logging.error('An exception occured while attempting to establish DB connection', e)
      raise e


'''
Fetch global DB connection
'''
def get_connection():
  global _connection
  if _connection is None: 
     raise Exception("Database connection is not intialized.") 
  return _connection


'''
Functionality to close our global DB connection
'''
def close_connection(): 
   global _connection
   if _connection:
      _connection.close() 
      _connection = None
   
   

