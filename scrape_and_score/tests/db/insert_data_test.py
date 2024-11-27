from unittest.mock import patch, MagicMock
from db import insert_data

@patch('db.insert_data.insert_player', return_value = None)
def test_insert_players_calls_insert_player_expected_number_of_times(mock_insert_player): 
   players = ['Player One', 'Player Two']
   
   insert_data.insert_players(players)
   
   assert mock_insert_player.call_count == 2