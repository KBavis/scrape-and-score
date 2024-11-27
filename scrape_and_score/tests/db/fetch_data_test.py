from unittest.mock import patch, MagicMock
from db import fetch_data

@patch('db.fetch_data.get_connection')
def test_fetch_all_teams_returns_teams(mock_get_connection):
    expected_teams = [
        {'team_id': 1, 'name': 'Team A', 'offense_rank': 10, 'defense_rank': 15},
        {'team_id': 2, 'name': 'Team B', 'offense_rank': 5, 'defense_rank': 8}
    ]
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = [
        (1, 'Team A', 10, 15),
        (2, 'Team B', 5, 8),
    ]
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_teams = fetch_data.fetch_all_teams() 
    
    assert actual_teams == expected_teams


@patch('db.fetch_data.get_connection', side_effect=Exception('Database connection failed'))
def test_fetch_all_throws_expected_exception(mock_get_connection):
    expected_exception = "Database connection failed"
    
    try :
        fetch_data.fetch_all_teams() 
    except Exception as e: 
        assert str(e) == expected_exception


@patch('db.fetch_data.get_connection')
def test_fetch_all_teams_returns_zero_teams_when_none_persisted(mock_get_connection):
    expected_teams = []
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_teams = fetch_data.fetch_all_teams() 
    
    assert actual_teams == expected_teams


@patch('db.fetch_data.get_connection')
def test_fetch_team_by_name_returns_expected_team(mock_get_connection):
    expected_team = {'team_id': 1, 'name': 'Team A', 'offense_rank': 10, 'defense_rank': 15}
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (1, 'Team A', 10, 15)
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_team = fetch_data.fetch_team_by_name('Team A') 
    
    assert actual_team == expected_team


@patch('db.fetch_data.get_connection')
def test_fetch_team_by_name_returns_no_team(mock_get_connection):
    expected_team = None
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = None
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_team = fetch_data.fetch_team_by_name('Team A') 
    
    assert actual_team == expected_team


@patch('db.fetch_data.get_connection', side_effect=Exception('Database connection failed'))
def test_fetch_team_by_name_throws_exception(mock_get_connection):
    try:
        actual_team = fetch_data.fetch_team_by_name('Team A') 
    except Exception as e:
        assert str(e) == 'Database connection failed'


@patch('db.fetch_data.get_connection')
def test_fetch_all_players_returns_expected_players(mock_get_connection):
    expected_players = [
        {'player_id': 1, 'team_id': 10, 'player_name': 'Anthony Gould', 'position': 'WR'},
        {'player_id': 2, 'team_id': 9, 'player_name': 'Anthony Richardson', 'position': 'QB'}
    ]
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = [
        (1, 10, 'Anthony Gould', 'WR'),
        (2, 9, 'Anthony Richardson', 'QB'),
    ]
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_players = fetch_data.fetch_all_players() 
    
    assert actual_players == expected_players

@patch('db.fetch_data.get_connection')
def test_fetch_all_players_returns_no_players_when_none_persisted(mock_get_connection):
    expected_players = []
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_players = fetch_data.fetch_all_players() 
    
    assert actual_players == expected_players


@patch('db.fetch_data.get_connection', side_effect=Exception('Database connection failed'))
def test_fetch_all_players_throws_exception(mock_get_connection):
    expected_msg = 'Database connection failed'
    
    try:
        actual_players = fetch_data.fetch_all_players() 
    except Exception as e:
        assert str(e) == expected_msg


@patch('db.fetch_data.get_connection')
def test_fetch_player_by_name_returns_expected_player(mock_get_connection):
    expected_player = {'player_id': 1, 'team_id': 10, 'name': 'Anthony Gould', 'position': 'WR'}
    
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (1, 10, 'Anthony Gould', 'WR')
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_player = fetch_data.fetch_player_by_name('Anthony Gould') 
    
    assert actual_player == expected_player 

@patch('db.fetch_data.get_connection')
def test_fetch_player_by_name_returns_no_player_if_none_persisted(mock_get_connection):
    expected_player = None
    
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = None
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_player = fetch_data.fetch_player_by_name('Anthony Gould') 
    
    assert actual_player == expected_player 

@patch('db.fetch_data.get_connection', side_effect=Exception('Database connection error'))
def test_fetch_player_by_name_throws_exception(mock_get_connection):
    try:
        fetch_data.fetch_player_by_name('Anthony Gould') 
    except Exception as e: 
        assert str(e) == 'Database connection error'