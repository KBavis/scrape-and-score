from scrape_and_score.workflows.utils import (
    is_player_demographics_persisted,
    is_player_records_persisted,
    are_team_seasonal_metrics_persisted,
    are_player_seasonal_metrics_persisted,
    add_stubbed_player_game_logs,
    generate_game_mapping,
    get_position_features,
    extract_file_contents,
    filter_completed_games,
    filter_duplicate_games
)
from datetime import datetime, timedelta
from unittest.mock import patch
from pytest import raises

SEASON = 2023
WEEK = 5


@patch("scrape_and_score.workflows.utils.get_count_player_demographics_records_for_season")
def test_is_player_demographics_persisted(mock_get_count):
    mock_get_count.return_value = 5
    assert is_player_demographics_persisted(SEASON) is True

    mock_get_count.return_value = 0
    assert is_player_demographics_persisted(SEASON) is False


@patch("scrape_and_score.workflows.utils.get_count_player_teams_records_for_season")
def test_is_player_records_persisted(mock_get_count):
    mock_get_count.return_value = 3
    assert is_player_records_persisted(SEASON) is True

    mock_get_count.return_value = 0
    assert is_player_records_persisted(SEASON) is False


@patch("scrape_and_score.workflows.utils.fetch_team_seasonal_metrics")
@patch("scrape_and_score.workflows.utils.team_service.get_team_id_by_name")
@patch("scrape_and_score.workflows.utils.props.get_config")
def test_are_team_seasonal_metrics_persisted(mock_config, mock_get_id, mock_fetch_metrics):
    mock_config.return_value = [{"name": "TeamA"}, {"name": "TeamB"}]

    # all metrics exist 
    mock_get_id.side_effect = [1, 2]
    mock_fetch_metrics.return_value = {"stat": 123}
    assert are_team_seasonal_metrics_persisted(SEASON) is True

    # one metric missing
    mock_get_id.side_effect = [1, 2] # reset side effect 
    def fetch_metrics_side_effect(team_id, season):
        return None if team_id == 1 else {"stat": 123}

    mock_fetch_metrics.side_effect = fetch_metrics_side_effect
    assert are_team_seasonal_metrics_persisted(SEASON) is False


@patch("scrape_and_score.workflows.utils.fetch_player_seasonal_metrics")
def test_are_player_seasonal_metrics_persisted(mock_fetch):
    mock_fetch.return_value = {"metrics": []}
    assert are_player_seasonal_metrics_persisted(SEASON) is True

    mock_fetch.return_value = None
    assert are_player_seasonal_metrics_persisted(SEASON) is False


@patch("scrape_and_score.workflows.utils.insert_upcoming_player_game_logs")
@patch("scrape_and_score.workflows.utils.fetch_player_teams_by_week_season_and_player_id")
@patch("scrape_and_score.workflows.utils.fetch_team_game_logs_by_week_and_season")
@patch("scrape_and_score.workflows.utils.fetch_player_game_log_by_pk")
def test_add_stubbed_player_game_logs(
    mock_fetch_log, mock_fetch_team_logs, mock_fetch_team, mock_insert
):
    mock_fetch_log.return_value = None  # simulate not yet persisted
    player_ids = [1, 2]
    mock_fetch_team_logs.return_value = [
        {"team_id": 100, "day": "Sunday", "home_team": "A", "opp": "B"}
    ]
    mock_fetch_team.side_effect = [100, 100]

    add_stubbed_player_game_logs(player_ids, WEEK, SEASON)
    assert mock_insert.called


def test_extract_file_contents(tmp_path):
    file = tmp_path / "QB_features_20230601010101.txt"
    contents = ["feature1", "feature2"]
    file.write_text("\n".join(contents))

    result = extract_file_contents(file.name, tmp_path)
    assert result == contents


def test_filter_completed_games():
    today = datetime.now().date()
    games = [
        {"game_date": today + timedelta(days=1), "team_ids": [1, 2]},
        {"game_date": today - timedelta(days=1), "team_ids": [3, 4]},
    ]
    filtered = filter_completed_games(games)
    assert len(filtered) == 1
    assert filtered[0]["team_ids"] == [1, 2]


def test_filter_duplicate_games():
    team_game_logs = [
        {"team_id": 1, "opp": 2},
        {"team_id": 2, "opp": 1},  # duplicate
        {"team_id": 3, "opp": 4}
    ]
    filtered = filter_duplicate_games(team_game_logs)
    assert len(filtered) == 2
    team_ids = [g["team_id"] for g in filtered]
    assert 1 in team_ids and 3 in team_ids


@patch("scrape_and_score.workflows.utils.fetch_players_corresponding_to_season_week_team")
@patch("scrape_and_score.workflows.utils.fetch_team_game_logs_by_week_and_season")
def test_generate_game_mapping(mock_fetch_team_logs, mock_fetch_players):
    mock_fetch_team_logs.return_value = [
        {"team_id": 1, "opp": 2, "game_date": datetime.now().date()},
        {"team_id": 2, "opp": 1, "game_date": datetime.now().date()},
    ]
    mock_fetch_players.return_value = [101, 102]

    mappings = generate_game_mapping(SEASON, WEEK)
    assert len(mappings) == 1
    assert "player_ids" in mappings[0]
    assert 101 in mappings[0]["player_ids"]


@patch("scrape_and_score.workflows.utils.extract_file_contents")
@patch("os.path.isfile")
@patch("os.listdir")
@patch("os.path.exists")
@patch("scrape_and_score.workflows.utils.datetime")
def test_get_position_features_success(mock_datetime, mock_exists, mock_listdir, mock_isfile, mock_extract):

    # arrange  
    mock_exists.return_value = True
    now = datetime(2025, 6, 7, 12, 0, 0)
    mock_datetime.now.return_value = now
    mock_datetime.strptime.side_effect = lambda s, f: datetime.strptime(s, f)
    
    # create files for all positions, with two files for QB to test closest time selection
    dt_old = now - timedelta(days=1)
    dt_new = now - timedelta(hours=1)
    files = [
        make_file_name("QB", dt_old),
        make_file_name("QB", dt_new),
        make_file_name("RB", now),
        make_file_name("WR", now),
        make_file_name("TE", now),
        "ignore.txt",  # irrelevant file
        "WR_wrongformat.csv"  # irrelevant format
    ]
    mock_listdir.return_value = files
    mock_isfile.return_value = True

    # mock extract_file_contents to return a dict with position name
    def extract_side_effect(file, dir):
        pos = file.split("_")[0]
        return [f"{pos}_feature1", f"{pos}_feature2"]
    mock_extract.side_effect = extract_side_effect

    features = get_position_features()
    
    # check that each position has features and QB features come from closest datetime file (dt_new)
    assert set(features.keys()) == {"QB", "RB", "WR", "TE"}
    assert features["QB"] == ["QB_feature1", "QB_feature2"]
    assert features["RB"] == ["RB_feature1", "RB_feature2"]
    assert features["WR"] == ["WR_feature1", "WR_feature2"]
    assert features["TE"] == ["TE_feature1", "TE_feature2"]

@patch("os.path.exists")
def test_directory_missing(mock_exists):
    mock_exists.return_value = False
    with raises(Exception, match="Please ensure that the directory"):
        get_position_features()

@patch("os.path.exists")
@patch("os.listdir")
def test_directory_empty(mock_listdir, mock_exists):
    mock_exists.return_value = True
    mock_listdir.return_value = []
    with raises(Exception, match="Please ensure that you first train"):
        get_position_features()

@patch("os.path.exists")
@patch("os.listdir")
@patch("os.path.isfile")
def test_missing_position_files(mock_isfile, mock_listdir, mock_exists):
    mock_exists.return_value = True
    mock_listdir.return_value = [
        "QB_features_20250101_120000.csv",
        "RB_features_20250101_120000.csv",
        # WR and TE missing
    ]
    mock_isfile.return_value = True
    with raises(Exception, match="Unable to retrieve features for positions:"):
        get_position_features()

@patch("os.path.exists")
@patch("os.listdir")
@patch("os.path.isfile")
@patch("scrape_and_score.workflows.utils.datetime")
def test_invalid_date_format_files(mock_datetime, mock_isfile, mock_listdir, mock_exists):
    mock_exists.return_value = True
    mock_datetime.now.return_value = datetime(2025, 6, 7, 12, 0, 0)
    mock_datetime.strptime.side_effect = lambda s, f: datetime.strptime(s, f)
    
    mock_listdir.return_value = [
        "QB_features_invaliddate.csv",
        "RB_features_20250101_120000.csv",
        "WR_features_20250101_120000.csv",
        "TE_features_20250101_120000.csv",
    ]
    mock_isfile.return_value = True

    with raises(Exception, match="Unable to retrieve features for positions:"):
        get_position_features()


# helper function for creating file names
def make_file_name(position, dt):
    return f"{position}_features_{dt.strftime('%Y%m%d')}_{dt.strftime('%H%M%S')}.csv"