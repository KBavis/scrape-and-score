from unittest.mock import patch, MagicMock
from db import fetch_data


from unittest.mock import patch, MagicMock
from db import fetch_data

@patch('db.fetch_data.get_connection')
def test_fetch_team_game_log_by_pk_when_game_log_not_persisted(mock_get_connection):
    expected_team_game_log = None
    mock_connection = MagicMock()

    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = None  # Simulate no game log found

    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    pk = {'team_id': 32, 'week': 2, 'year': 2024}
    actual_team_game_log = fetch_data.fetch_team_game_log_by_pk(pk)

    assert actual_team_game_log == expected_team_game_log


@patch('db.fetch_data.get_connection')
def test_fetch_player_game_log_by_pk_when_game_log_not_persisted(mock_get_connection):
    expected_player_game_log = None
    mock_connection = MagicMock()

    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = None  # Simulate no game log found

    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    pk = {'player_id': 527, 'week': 2, 'year': 2024}
    actual_player_game_log = fetch_data.fetch_player_game_log_by_pk(pk)

    assert actual_player_game_log == expected_player_game_log


@patch('db.fetch_data.get_connection')
def test_fetch_team_game_log_by_pk_when_game_log_persisted(mock_get_connection):
    expected_team_game_log = {
        'team_id': 32,
        'week': 2,
        'day': 'Mon',
        'year': 2024,
        'rest_days': 3,
        'home_team': True,
        'distance_traveled': 150.0,
        'opp': 45,
        'result': 'W',
        'points_for': 30,
        'points_allowed': 20,
        'tot_yds': 400,
        'pass_yds': 250,
        'rush_yds': 150,
        'opp_tot_yds': 300,
        'opp_pass_yds': 200,
        'opp_rush_yds': 100
    }
    mock_connection = MagicMock()

    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (
        32, 2, 'Mon', 2024, 3, True, 150.0, 45, 'W', 30, 20,
        400, 250, 150, 300, 200, 100
    )

    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    pk = {'team_id': 32, 'week': 2, 'year': 2024}
    actual_team_game_log = fetch_data.fetch_team_game_log_by_pk(pk)

    assert actual_team_game_log == expected_team_game_log


@patch('db.fetch_data.get_connection')
def test_fetch_player_game_log_by_pk_when_game_log_persisted(mock_get_connection):
    expected_player_game_log = {
        'player_id': 527,
        'week': 2,
        'day': 16,
        'year': 2024,
        'home_team': True,
        'opp': 45,
        'result': 'W',
        'points_for': 30,
        'points_allowed': 20,
        'completions': 25,
        'attempts': 35,
        'pass_yd': 300,
        'pass_td': 3,
        'interceptions': 1,
        'rating': 105.5,
        'sacked': 2,
        'rush_att': 10,
        'rush_yds': 50,
        'rush_tds': 1,
        'tgt': 5,
        'rec': 4,
        'rec_yd': 60,
        'rec_td': 1,
        'snap_pct': 85.0
    }
    mock_connection = MagicMock()

    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (
        527, 2, 16, 2024, True, 45, 'W', 30, 20, 25, 35,
        300, 3, 1, 105.5, 2, 10, 50, 1, 5, 4, 60, 1, 85.0
    )

    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    pk = {'player_id': 527, 'week': 2, 'year': 2024}
    actual_player_game_log = fetch_data.fetch_player_game_log_by_pk(pk)

    assert actual_player_game_log == expected_player_game_log


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
        fetch_data.fetch_team_by_name('Team A') 
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



@patch('db.fetch_data.get_connection', side_effect=Exception('Database connection error'))
def test_fetch_one_team_game_log_throws_exception(mock_get_connection):
    try:
        fetch_data.fetch_one_team_game_log()
    except Exception as e:
        assert str(e) == 'Database connection error'

@patch('db.fetch_data.get_connection')
def test_fetch_one_team_game_log_returns_expected_game_log(mock_get_connection):
    expected_game_log = {
        'team_id': 2,
        'week': 14,
        'year': 2024
    }
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (2, 14, 'Arbitrary', 2024)
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_game_log = fetch_data.fetch_one_team_game_log()
    
    assert expected_game_log == actual_game_log


@patch('db.fetch_data.get_connection')
def test_fetch_one_team_game_log_calls_expected_functions(mock_get_connection): 
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (2, 'Value', 'Arbitrary', 'Test')
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    fetch_data.fetch_one_team_game_log()
    
    mock_get_connection.assert_called_once()


@patch('db.fetch_data.get_connection')
def test_fetch_one_team_game_log_executes_expected_sql(mock_get_connection):
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (2, 'Value', 'Arbitrary', 'Test')
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    fetch_data.fetch_one_team_game_log()
    
    mock_cursor.execute.assert_called_once_with('SELECT * FROM team_game_log FETCH FIRST 1 ROW ONLY')

@patch('db.fetch_data.get_connection')
def test_fetch_one_team_game_log_when_no_records_persisted(mock_get_connection):
    expected_team_game_log = None
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = None # ensure nothing returned
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_team_game_log = fetch_data.fetch_one_team_game_log()
    
    assert actual_team_game_log == expected_team_game_log


@patch('db.fetch_data.get_connection', side_effect=Exception('Database connection error'))
def test_fetch_one_player_game_log_throws_exception(mock_get_connection):
    try:
        fetch_data.fetch_one_player_game_log()
    except Exception as e:
        assert str(e) == 'Database connection error'
        

@patch('db.fetch_data.get_connection')
def test_fetch_one_player_game_log_returns_expected_game_log(mock_get_connection):
    expected_game_log = {
        'player_id': 2,
        'week': 14,
        'year': 2024
    }
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (2, 14, 'Arbitrary', 2024)
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_game_log = fetch_data.fetch_one_player_game_log()
    
    assert expected_game_log == actual_game_log


@patch('db.fetch_data.get_connection')
def test_fetch_one_player_game_log_calls_expected_functions(mock_get_connection): 
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (2, 'Value', 'Arbitrary', 2023)
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    fetch_data.fetch_one_player_game_log()
    
    mock_get_connection.assert_called_once()


@patch('db.fetch_data.get_connection')
def test_fetch_one_player_game_log_executes_expected_sql(mock_get_connection):
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (2, 'Value', 'Arbitrary', 2023)
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    fetch_data.fetch_one_player_game_log()
    
    mock_cursor.execute.assert_called_once_with('SELECT * FROM player_game_log FETCH FIRST 1 ROW ONLY')

@patch('db.fetch_data.get_connection')
def test_fetch_one_player_game_log_when_no_records_persisted(mock_get_connection):
    expected_player_game_log = None
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = None # ensure nothing returned
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_player_game_log = fetch_data.fetch_one_player_game_log()
    
    assert actual_player_game_log == expected_player_game_log


@patch('db.fetch_data.get_connection')
def test_fetch_all_player_game_logs_for_recent_week_returns_expected_list(mock_get_connection):
    expected_player_game_log = {
        'player_id': 527,
        'week': 2,
        'day': 16,
        'year': 2024,
        'home_team': True,
        'opp': 45,
        'result': 'W',
        'points_for': 30,
        'points_allowed': 20,
        'completions': 25,
        'attempts': 35,
        'pass_yd': 300,
        'pass_td': 3,
        'interceptions': 1,
        'rating': 105.5,
        'sacked': 2,
        'rush_att': 10,
        'rush_yds': 50,
        'rush_tds': 1,
        'tgt': 5,
        'rec': 4,
        'rec_yd': 60,
        'rec_td': 1,
        'snap_pct': 85.0
    }
    expected_game_logs = [expected_player_game_log]
    
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = [(527, 2, 16, 2024, True, 45, 'W', 30, 20, 25, 35, 300, 3, 1, 105.5, 2, 10, 50, 1, 5, 4, 60, 1, 85.0)]
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_game_logs = fetch_data.fetch_all_player_game_logs_for_recent_week(2024)
    
    assert actual_game_logs == expected_game_logs

@patch('db.fetch_data.get_connection')
def test_fetch_all_player_game_logs_for_recent_week_executes_expected_sql(mock_get_connection): 
    
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = None
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    fetch_data.fetch_all_player_game_logs_for_recent_week(2024)
    
    mock_cursor.execute.assert_called_once_with(
        'SELECT * FROM player_game_log WHERE year=%s AND week=(SELECT MAX(week) FROM player_game_log)',
        (2024)
    )

@patch('db.fetch_data.get_connection')
def test_fetch_all_player_game_logs_for_recent_week_calls_expected_functions(mock_get_connection): 
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = None
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    fetch_data.fetch_all_player_game_logs_for_recent_week(2024)
    
    mock_get_connection.assert_called_once()


@patch('db.fetch_data.get_connection')
def test_fetch_all_player_game_logs_for_given_year_returns_expected_list(mock_get_connection):
    expected_player_game_log = {
        'player_id': 527,
        'week': 2,
        'day': 16,
        'year': 2024,
        'home_team': True,
        'opp': 45,
        'result': 'W',
        'points_for': 30,
        'points_allowed': 20,
        'completions': 25,
        'attempts': 35,
        'pass_yd': 300,
        'pass_td': 3,
        'interceptions': 1,
        'rating': 105.5,
        'sacked': 2,
        'rush_att': 10,
        'rush_yds': 50,
        'rush_tds': 1,
        'tgt': 5,
        'rec': 4,
        'rec_yd': 60,
        'rec_td': 1,
        'snap_pct': 85.0
    }
    expected_game_logs = [expected_player_game_log]
    
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = [(527, 2, 16, 2024, True, 45, 'W', 30, 20, 25, 35, 300, 3, 1, 105.5, 2, 10, 50, 1, 5, 4, 60, 1, 85.0)]
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    actual_game_logs = fetch_data.fetch_all_player_game_logs_for_given_year(2024)
    
    assert actual_game_logs == expected_game_logs

@patch('db.fetch_data.get_connection')
def test_fetch_all_player_game_logs_for_given_year_executes_expected_sql(mock_get_connection): 
    
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = None
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    fetch_data.fetch_all_player_game_logs_for_given_year(2024)
    
    mock_cursor.execute.assert_called_once_with(
        'SELECT * FROM player_game_log WHERE year=%s',
        (2024)
    )

@patch('db.fetch_data.get_connection')
def test_fetch_all_player_game_logs_for_given_year_calls_expected_functions(mock_get_connection): 
    mock_connection = MagicMock() 
    
    mock_cursor = MagicMock() 
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = None
    
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor 
    mock_get_connection.return_value = mock_connection
    
    fetch_data.fetch_all_player_game_logs_for_given_year(2024)
    
    mock_get_connection.assert_called_once()