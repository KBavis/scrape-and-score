from unittest.mock import patch
from db import util

teams = ['Indianapolis Colts', 'Atlanta Falcons']
players = ['Calvin Ridley', 'Jerimiah Brown']

@patch('db.util.fetch_all_teams', return_value = teams)
def test_is_team_empty_returns_false(mock_fetch_all_teams): 
   assert util.is_team_empty() == False


@patch('db.util.fetch_all_teams', return_value = None)
def test_is_team_empty_returns_true(mock_fetch_all_teams): 
   assert util.is_team_empty() == True


@patch('db.util.fetch_all_teams', return_value = teams)
def test_is_team_empty_calls_expected_functions(mock_fetch_all_teams): 
   util.is_team_empty()
   mock_fetch_all_teams.assert_called_once()


@patch('db.util.fetch_all_players', return_value = players)
def test_is_player_empty_returns_false(mock_fetch_all_players): 
   assert util.is_player_empty() == False


@patch('db.util.fetch_all_players', return_value = None)
def test_is_player_empty_returns_true(mock_fetch_all_players): 
   assert util.is_player_empty() == True


@patch('db.util.fetch_all_players', return_value = players)
def test_is_player_empty_calls_expected_functions(mock_fetch_all_players): 
   util.is_player_empty()
   mock_fetch_all_players.assert_called_once()
