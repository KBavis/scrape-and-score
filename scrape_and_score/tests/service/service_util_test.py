from service import service_util
from unittest.mock import patch

def test_extract_day_from_date_returns_expected_day():
   expected_day = '30'
   date = '2024-09-30'
   assert expected_day == service_util.extract_day_from_date(date)

def test_extract_year_from_date_returns_expected_year():
   expected_year = '2024'
   date = '2024-09-30'
   assert expected_year == service_util.extract_year_from_date(date)

def test_extract_home_team_from_game_location_returns_expected_flag_when_not_home():
   expected_flag = 'F'
   game_location = '@'
   assert expected_flag == service_util.extract_home_team_from_game_location(game_location)
   
def test_extract_home_team_from_game_location_returns_expected_flag_when_home():
   expected_flag = 'T'
   game_location = 'Random'
   assert expected_flag == service_util.extract_home_team_from_game_location(game_location)

@patch('service.service_util.props.get_config') 
def test_get_team_name_by_pfr_acronym_returns_expected_team_when_found(mock_get_config):
   opp = 'IND'
   expected_name = 'Colts'
   mock_get_config.return_value = [{'name': expected_name, 'pfr_acronym': 'IND'}, {'name': 'Random', 'pfr_acronym': 'RAN'}]
   
   actual_name = service_util.get_team_name_by_pfr_acronym(opp)
   
   assert actual_name == expected_name
   
   

@patch('service.service_util.props.get_config') 
def test_get_team_name_by_pfr_acronym_returns_none_when_no_team_found(mock_get_config):
   opp = 'NA' # ensure no acronym within list of teams exists 
   mock_get_config.return_value = [{'name': 'Colts', 'pfr_acronym': 'IND'}, {'name': 'Random', 'pfr_acronym': 'RAN'}]
   
   actual_name = service_util.get_team_name_by_pfr_acronym(opp)
   
   assert actual_name == None
   
def test_get_game_log_year_when_week_eighteen():
   year = 2024
   assert service_util.get_game_log_year('18', year) == 2025

def test_get_game_log_year_when_not_week_eighteen(): 
   year = 2024
   assert service_util.get_game_log_year('16', year) == year