"""
Functionality to mock os.getenv() 

Args:
   key (str): key to fetch 
Returns:
   mocked_value (str): mocked value corresponding to key
"""


def mock_get_env(key: str):
    if key == "DB_NAME":
        return "db_name"
    elif key == "DB_PASS":
        return "db_pass"
    elif key == "DB_PORT":
        return "db_port"
    elif key == "DB_USER":
        return "db_user"
    elif key == "DB_HOST":
        return "db_host"
