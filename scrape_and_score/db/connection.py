from psycopg2 import connect
from dotenv import load_dotenv
import os
import logging

_connection = None

"""
Intialize a global DB connection 
"""


def init():
    global _connection
    load_dotenv()

    try:
        db_name = os.getenv("DB_NAME")
        password = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        user = os.getenv("DB_USER")

        if not all([db_name, password, host, port, user]):
            raise ValueError(
                "One or more required environment variables for establishing a db connection are missing."
            )

        _connection = connect(
            database=db_name, user=user, password=password, host=host, port=port
        )

    except Exception as e:
        logging.error(
            f"An exception occured while attempting to establish DB connection: {e}"
        )
        raise e


"""
Fetch global DB connection
"""


def get_connection():
    global _connection

    if _connection is None:
        raise Exception("Database connection is not intialized.")
    return _connection


"""
Functionality to close our global DB connection
"""


def close_connection():
    global _connection
    if _connection:
        _connection.close()
        _connection = None
