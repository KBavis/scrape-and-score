from service import player_service 
from unittest.mock import patch 

@patch('service.player_service.fetch_all_players')
def test_get_all_players_returns_expected_players_when_persisted(mock_fetch_all_players):
   expected_players = [{'name': 'Anthony Richardson', 'position': 'QB'}, {'name': 'Jonathon Taylor', 'position': 'RB'}]
   mock_fetch_all_players.return_value = expected_players
   assert player_service.get_all_players() == expected_players

@patch('service.player_service.fetch_all_players', return_value = [])
def test_get_all_players_returns_expected_players_when_none_persisted(mock_fetch_all_players):
   assert player_service.get_all_players() == []