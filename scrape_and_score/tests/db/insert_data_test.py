from unittest.mock import patch, MagicMock
from db import insert_data
from random import random 

@patch('db.insert_data.insert_player', return_value = None)
def test_insert_players_calls_insert_player_expected_number_of_times(mock_insert_player): 
   players = ['Player One', 'Player Two']
   
   insert_data.insert_players(players)
   
   assert mock_insert_player.call_count == 2


@patch('db.insert_data.insert_player', side_effect = Exception('Unable to insert the following player: Kyler Murray'))
def test_insert_players_when_exceptions_occurs(mock_insert_player): 
   players = ['Player One', 'Player Two']
   
   try:
      insert_data.insert_players(players)
   except Exception as e: 
      assert str(e) == "Unable to insert the following player: Kyler Murray"
      

@patch('db.insert_data.get_connection', side_effect = Exception('Unable to fetch database connection'))
def test_insert_player_when_exception_occurs(mock_get_connection): 
   # set up test data & mocks 
   test_player = {'player_name': 'Anthony Richardson', 'position': 'QB', 'team_id': 5}
   
   try: 
      insert_data.insert_player(test_player)
   except Exception as e: 
      assert str(e) == 'Unable to fetch database connection'
   

@patch('db.insert_data.get_connection')
def test_insert_player_calls_expected_functions(mock_get_connection): 
   # set up test data & mocks 
   test_player = {'player_name': 'Anthony Richardson', 'position': 'QB', 'team_id': 5}
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   
   mock_cursor = MagicMock() 
   mock_cursor.execute.return_value = None 
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
   
   mock_get_connection.return_value = mock_connection
   
   insert_data.insert_player(test_player)
   
   mock_get_connection.assert_called_once() 

@patch('db.insert_data.get_connection')
def test_insert_player_calls_expected_functions(mock_get_connection): 
   # set up test data & mocks 
   test_player = {'player_name': 'Anthony Richardson', 'position': 'QB', 'team_id': 5}
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   
   mock_cursor = MagicMock() 
   mock_cursor.execute.return_value = None 
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
   
   mock_get_connection.return_value = mock_connection
   
   insert_data.insert_player(test_player)
   
   mock_cursor.execute.assert_called_once_with(
      "\n      INSERT INTO player (team_id, name, position) \n      VALUES (%s, %s, %s)\n   ",
      (test_player['team_id'], test_player['player_name'], test_player['position'])
   )
   mock_connection.commit.assert_called_once()


@patch('db.insert_data.insert_team', return_value = random())
def test_insert_teams_calls_insert_team_expected_number_of_times(mock_insert_team): 
   teams = ['Team One', 'Team Two']
   
   insert_data.insert_teams(teams)
   
   assert mock_insert_team.call_count == 2


@patch('db.insert_data.insert_team', side_effect = Exception('Unable to insert the following team: Team One'))
def test_insert_teams_when_exceptions_occurs(mock_insert_team): 
   teams = ['Team One', 'Team Two']
   
   try:
      insert_data.insert_teams(teams)
   except Exception as e: 
      assert str(e) == "Unable to insert the following team: Team One"


@patch('db.insert_data.insert_team', return_value = 1)
def test_insert_teams_returns_expected_list(mock_insert_team): 
   teams = ['Team One', 'Team Two']
   expected_list = [{'team_id': 1, 'name': 'Team One'}, {'team_id': 1, 'name': 'Team Two'}]
   
   actual_list = insert_data.insert_teams(teams)
   
   assert expected_list == actual_list



@patch('db.insert_data.get_connection', side_effect = Exception('Unable to fetch database connection'))
def test_insert_team_when_exception_occurs(mock_get_connection): 
   # set up test data & mocks 
   test_team = 'Indianapolis Colts'
   
   try: 
      insert_data.insert_team(test_team)
   except Exception as e: 
      assert str(e) == 'Unable to fetch database connection'
   

@patch('db.insert_data.get_connection')
def test_insert_team_calls_expected_functions(mock_get_connection): 
   # set up test data & mocks 
   test_team = 'Indianapolis Colts'
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   
   mock_cursor = MagicMock() 
   mock_cursor.execute.return_value = None 
   mock_cursor.fetchone.return_value = [1]
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
   
   mock_get_connection.return_value = mock_connection
   
   insert_data.insert_team(test_team)
   
   mock_get_connection.assert_called_once() 

@patch('db.insert_data.get_connection')
def test_insert_team_returns_expected_team_id(mock_get_connection): 
   # set up test data & mocks 
   test_team = 'Indianapolis Colts'
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   
   mock_cursor = MagicMock() 
   mock_cursor.execute.return_value = None 
   mock_cursor.fetchone.return_value = [1]
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
   
   mock_get_connection.return_value = mock_connection
   
   team_id = insert_data.insert_team(test_team)
   
   assert team_id == 1

@patch('db.insert_data.get_connection')
def test_insert_team_attempts_to_insert_expected_team(mock_get_connection): 
   # set up test data & mocks 
   test_team = 'Indianapolis Colts'
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   
   mock_cursor = MagicMock() 
   mock_cursor.execute.return_value = None 
   mock_cursor.fetchone.return_value = [random()]
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
   
   mock_get_connection.return_value = mock_connection
   
   insert_data.insert_team(test_team)
   
   mock_cursor.execute.assert_called_once_with(
      "INSERT INTO team (name) VALUES (%s) RETURNING team_id", (test_team,)
   )


@patch('db.insert_data.get_connection', side_effect = Exception('Database connection failed'))
def test_insert_team_game_logs_when_exception_occurs(mock_get_connection):
   try:
      insert_data.insert_team_game_logs([(1,2), (3,4)])
   except Exception as e:
      assert str(e) == 'Database connection failed'
   
   

@patch('db.insert_data.get_connection')
def test_insert_team_game_logs_calls_expected_functions(mock_get_connection):
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   mock_cursor = MagicMock() 
   mock_cursor.executemany.return_value = None 
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
   mock_get_connection.return_value = mock_connection
   
   insert_data.insert_team_game_logs([(1,2,3), (4,5,6)])
   
   mock_get_connection.assert_called_once()
   mock_cursor.executemany.assert_called_once()
   mock_connection.commit.assert_called_once()

@patch('db.insert_data.get_connection', side_effect = Exception('Database connection failed'))
def test_insert_rb_player_game_logs_when_exception_occurs(mock_get_connection):
   try:
      insert_data.insert_rb_player_game_logs([(1,2), (3,4)])
   except Exception as e:
      assert str(e) == 'Database connection failed'
   
   

@patch('db.insert_data.get_connection')
def test_insert_rb_player_game_logs_calls_expected_functions(mock_get_connection):
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   mock_cursor = MagicMock() 
   mock_cursor.executemany.return_value = None 
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
   mock_get_connection.return_value = mock_connection
   
   insert_data.insert_rb_player_game_logs([(1,2,3), (4,5,6)])
   
   mock_get_connection.assert_called_once()
   mock_cursor.executemany.assert_called_once()
   mock_connection.commit.assert_called_once()

@patch('db.insert_data.get_connection', side_effect = Exception('Database connection failed'))
def test_insert_qb_player_game_logs_when_exception_occurs(mock_get_connection):
   try:
      insert_data.insert_qb_player_game_logs([(1,2), (3,4)])
   except Exception as e:
      assert str(e) == 'Database connection failed'
   
   

@patch('db.insert_data.get_connection')
def test_insert_qb_player_game_logs_calls_expected_functions(mock_get_connection):
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   mock_cursor = MagicMock() 
   mock_cursor.executemany.return_value = None 
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
   mock_get_connection.return_value = mock_connection
   
   insert_data.insert_qb_player_game_logs([(1,2,3), (4,5,6)])
   
   mock_get_connection.assert_called_once()
   mock_cursor.executemany.assert_called_once()
   mock_connection.commit.assert_called_once()


@patch('db.insert_data.get_connection', side_effect = Exception('Database connection failed'))
def test_insert_wr_or_te_player_game_logs_when_exception_occurs(mock_get_connection):
   try:
      insert_data.insert_wr_or_te_player_game_logs([(1,2), (3,4)])
   except Exception as e:
      assert str(e) == 'Database connection failed'
   
   

@patch('db.insert_data.get_connection')
def test_insert_wr_or_te_player_game_logs_calls_expected_functions(mock_get_connection):
   mock_connection = MagicMock() 
   mock_connection.commit.return_value = None 
   mock_cursor = MagicMock() 
   mock_cursor.executemany.return_value = None 
   mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
   mock_get_connection.return_value = mock_connection
   
   insert_data.insert_wr_or_te_player_game_logs([(1,2,3), (4,5,6)])
   
   mock_get_connection.assert_called_once()
   mock_cursor.executemany.assert_called_once()
   mock_connection.commit.assert_called_once()