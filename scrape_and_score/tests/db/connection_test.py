from unittest.mock import patch, MagicMock
from db import connection
from db_helper import mock_get_env
import pytest


@patch('db.connection.connect')
@patch('db.connection.os.getenv', side_effect=mock_get_env)
def test_init_successfully_establishes_connection(mock_os_getenv, mock_connect):
    # Arrange
    mock_connect.return_value = MagicMock()

    # Act
    connection.init()

    # Assert
    mock_connect.assert_called_once_with(
        database='db_name',
        user='db_user',
        password='db_pass',
        host='db_host',
        port='db_port'
    )
    assert mock_os_getenv.call_count == 5


@patch('db.connection.connect', side_effect=Exception("Connection failed")) 
@patch('db.connection.os.getenv', side_effect=mock_get_env)
def test_init_throws_exception(mock_os_getenv, mock_connect):
    
    with pytest.raises(Exception) as e: 
        connection.init()
    
    assert str(e.value) == "Connection failed"  
    
    mock_connect.assert_called_once()


@patch('db.connection.connect')
@patch('db.connection.os.getenv', return_value = None)
def test_init_throws_value_error(mock_os_getenv, mock_connect):
    
    with pytest.raises(ValueError) as e: 
        connection.init()
    
    assert str(e.value) == "One or more required environment variables for establishing a db connection are missing."  
    
    mock_connect.assert_not_called()


@patch('db.connection.connect')
@patch('db.connection.os.getenv', side_effect=mock_get_env)
def test_init_calls_expected_functions(mock_os_getenv, mock_connect):
    connection.init()

    assert mock_os_getenv.call_count == 5
    mock_connect.assert_called_once()

@patch('db.connection._connection', None)
def test_get_connection_raises_exception():
    try:
        connection.get_connection()
    except Exception as e:
        assert str(e) == "Database connection is not intialized."


@patch('db.connection._connection', MagicMock())
def test_get_connection_returns_expected_connection():
    conn = connection.get_connection()

    assert conn is not None


@patch('db.connection._connection', new_callable=MagicMock)
def test_close_connection_calls_close(mock_connection):
    mock_connection.close = MagicMock()

    connection.close_connection()

    mock_connection.close.assert_called_once()
    assert connection._connection is None
