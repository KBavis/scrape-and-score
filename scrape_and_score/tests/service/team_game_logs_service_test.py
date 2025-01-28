from service import team_game_logs_service
from unittest.mock import patch
import pandas as pd


@patch(
    "service.team_game_logs_service.fetch_data.fetch_team_game_log_by_pk",
    return_value={"team_id": 12, "week": 12, "year": 2024},
)
def test_is_game_log_persisted_returns_true(mock_fetch_game_log):
    game_log_pk = {"player_id": 12, "week": 12, "year": 2024}
    assert team_game_logs_service.is_game_log_persisted(game_log_pk) == True


@patch(
    "service.team_game_logs_service.fetch_data.fetch_team_game_log_by_pk",
    return_value=None,
)
def test_is_game_log_persisted_returns_false(mock_fetch_game_log):
    game_log_pk = {"player_id": 12, "week": 12, "year": 2024}
    assert team_game_logs_service.is_game_log_persisted(game_log_pk) == False


@patch(
    "service.team_game_logs_service.fetch_data.fetch_team_game_log_by_pk",
    return_value=None,
)
def test_is_game_log_persisted_calls_expected_functions(mock_fetch_game_log):
    team_game_logs_service.is_game_log_persisted({"team_id": 12})
    mock_fetch_game_log.assert_called_once_with({"team_id": 12})


@patch("service.team_game_logs_service.is_game_log_persisted", return_value=True)
@patch(
    "service.team_game_logs_service.service_util.get_game_log_year", return_value=2024
)
@patch("service.team_game_logs_service.get_team_id_by_name", return_value=14)
def test_remove_previously_inserted_games_skips_removal_when_not_recent_games(
    mock_get_id, mock_get_game_log_year, mock_is_game_log_persisted
):
    data = {
        "week": [1, 2],
        "day": ["Sun", "Sun"],
        "rest_days": [7, 1],
        "home_team": ["T", "F"],
        "distance_traveled": [500, 400],
        "opp": ["IND", "LAC"],
        "result": ["W", "W"],
        "points_for": [24, 24],
        "points_allowed": [10, 10],
        "tot_yds": [350, 240],
        "pass_yds": [250, 240],
        "rush_yds": [100, 201],
        "opp_tot_yds": [300, 212],
        "opp_pass_yds": [200, 232],
        "opp_rush_yds": [100, 232],
    }
    team_metrics = [{"team_name": "My Team", "team_metrics": pd.DataFrame(data=data)}]
    team_game_logs_service.remove_previously_inserted_game_logs(team_metrics, 2024, [])
    assert (
        mock_is_game_log_persisted.call_count == 0
    )  # skips when not only recent games


@patch("service.team_game_logs_service.is_game_log_persisted", return_value=True)
@patch(
    "service.team_game_logs_service.service_util.get_game_log_year", return_value=2024
)
@patch("service.team_game_logs_service.get_team_id_by_name", return_value=14)
def test_remove_previously_inserted_games_calls_expected_functions(
    mock_get_id, mock_get_game_log_year, mock_is_game_log_persisted
):
    data = {
        "week": [1],
        "day": ["Sun"],
        "rest_days": [7],
        "home_team": [1],
        "distance_traveled": [500],
        "opp": ["IND"],
        "result": ["W 24-10"],
        "points_for": [24],
        "points_allowed": [10],
        "tot_yds": [350],
        "pass_yds": [250],
        "rush_yds": [100],
        "opp_tot_yds": [300],
        "opp_pass_yds": [200],
        "opp_rush_yds": [100],
    }
    team_metrics = [{"team_name": "My Team", "team_metrics": pd.DataFrame(data=data)}]

    team_game_logs_service.remove_previously_inserted_game_logs(team_metrics, 2024, [])

    mock_is_game_log_persisted.assert_called_once_with(
        {"team_id": 14, "week": "1", "year": 2024}
    )
    mock_get_game_log_year.assert_called_once()
    mock_get_id.assert_called_once()


@patch(
    "service.team_game_logs_service.is_game_log_persisted", return_value=True
)  # ensure game log persisted
@patch(
    "service.team_game_logs_service.service_util.get_game_log_year", return_value=2024
)
@patch("service.team_game_logs_service.get_team_id_by_name", return_value=14)
def test_remove_previously_inserted_games__deletes_entries_from_player_metrics(
    mock_get_id, mock_get_game_log_year, mock_is_game_log_persisted
):
    data = {"week": [2]}
    team_metrics = [{"team_name": "My Team", "team_metrics": pd.DataFrame(data=data)}]

    team_game_logs_service.remove_previously_inserted_game_logs(team_metrics, 2024, [])

    assert len(team_metrics) == 0


@patch(
    "service.team_game_logs_service.is_game_log_persisted", return_value=False
)  # ensure game log NOT persisted
@patch(
    "service.team_game_logs_service.service_util.get_game_log_year", return_value=2024
)
@patch("service.team_game_logs_service.get_team_id_by_name", return_value=14)
def test_remove_previously_inserted_games__deletes_entries_from_player_metrics(
    mock_get_id, mock_get_game_log_year, mock_is_game_log_persisted
):
    data = {
        "week": [1],
        "day": ["Sun"],
        "rest_days": [7],
        "home_team": [1],
        "distance_traveled": [500],
        "opp": ["IND"],
        "result": ["W 24-10"],
        "points_for": [24],
        "points_allowed": [10],
        "tot_yds": [350],
        "pass_yds": [250],
        "rush_yds": [100],
        "opp_tot_yds": [300],
        "opp_pass_yds": [200],
        "opp_rush_yds": [100],
    }
    team_metrics = [{"team_name": "My Team", "team_metrics": pd.DataFrame(data=data)}]

    team_game_logs_service.remove_previously_inserted_game_logs(team_metrics, 2024, [])

    assert len(team_metrics) == 1


@patch("service.team_game_logs_service.get_team_id_by_name")
@patch("service.team_game_logs_service.get_team_log_tuples")
@patch("service.team_game_logs_service.insert_data.insert_team_game_logs")
@patch("service.team_game_logs_service.props.get_config")
@patch(
    "service.team_game_logs_service.remove_previously_inserted_game_logs",
    return_value=None,
)
def test_insert_multiple_teams_game_logs_skips_insert(
    mock_remove,
    mock_get_config,
    mock_insert_team_game_logs,
    mock_get_team_log_tuples,
    mock_get_team_id_by_name,
):
    team_game_logs_service.insert_multiple_teams_game_logs([], [])
    mock_remove.assert_called_once()
    mock_get_config.assert_called_once()
    assert mock_get_team_id_by_name.call_count == 0
    assert mock_get_team_log_tuples.call_count == 0
    assert mock_insert_team_game_logs.call_count == 0


@patch("service.team_game_logs_service.get_team_id_by_name")
@patch("service.team_game_logs_service.get_team_log_tuples")
@patch("service.team_game_logs_service.insert_data.insert_team_game_logs")
@patch("service.team_game_logs_service.props.get_config")
@patch(
    "service.team_game_logs_service.remove_previously_inserted_game_logs",
    return_value=None,
)
def test_insert_multiple_teams_game_logs_calls_expected_functions(
    mock_remove,
    mock_get_config,
    mock_insert_team_game_logs,
    mock_get_team_log_tuples,
    mock_get_team_id_by_name,
):
    df = pd.DataFrame(data=[{"random"}])
    team_metrics = [{"team_name": "Indianapolis Colts", "team_metrics": df}]

    mock_get_config.return_value = 2024
    mock_get_team_id_by_name.return_value = 12
    mock_insert_team_game_logs.return_value = None
    mock_get_team_log_tuples.return_value = [(1, 2, 3), (4, 5, 6)]

    team_game_logs_service.insert_multiple_teams_game_logs(team_metrics, [])

    mock_get_config.assert_called_once()
    mock_get_team_log_tuples.assert_called_once()
    mock_get_team_id_by_name.assert_called_once()
    mock_insert_team_game_logs.assert_called_once()


@patch("service.team_game_logs_service.team_service.get_team_id_by_name")
@patch("service.team_game_logs_service.service_util.get_game_log_year")
def test_get_team_log_tuples_returns_expected_tuples(
    mock_get_game_log_year, mock_get_team_id
):
    data = {
        "week": [1],
        "day": ["Sun"],
        "rest_days": [7],
        "home_team": [1],
        "distance_traveled": [500],
        "opp": ["IND"],
        "result": ["W 24-10"],
        "points_for": [24],
        "points_allowed": [10],
        "tot_yds": [350],
        "pass_yds": [250],
        "rush_yds": [100],
        "opp_tot_yds": [300],
        "opp_pass_yds": [200],
        "opp_rush_yds": [100],
    }
    team_id = 12
    df = pd.DataFrame(data)
    mock_get_team_id.return_value = 2
    mock_get_game_log_year.return_value = 2023

    expected_tuples = [
        (
            team_id,
            1,
            "Sun",
            2023,
            7,
            1,
            500,
            2,
            "W 24-10",
            24,
            10,
            350,
            250,
            100,
            300,
            200,
            100,
        )
    ]

    actual_tuples = team_game_logs_service.get_team_log_tuples(df, team_id, 2024)

    assert actual_tuples == expected_tuples


@patch("service.team_game_logs_service.team_service.get_team_id_by_name")
@patch("service.team_game_logs_service.service_util.get_game_log_year")
def test_get_team_log_tuples_calls_expected_functions(
    mock_get_game_log_year, mock_get_team_id
):
    data = {
        "week": [1],
        "day": ["Sun"],
        "rest_days": [7],
        "home_team": [1],
        "distance_traveled": [500],
        "opp": ["IND"],
        "result": ["W 24-10"],
        "points_for": [24],
        "points_allowed": [10],
        "tot_yds": [350],
        "pass_yds": [250],
        "rush_yds": [100],
        "opp_tot_yds": [300],
        "opp_pass_yds": [200],
        "opp_rush_yds": [100],
    }
    team_id = 12
    df = pd.DataFrame(data)

    team_game_logs_service.get_team_log_tuples(df, team_id, 2024)

    mock_get_game_log_year.assert_called_once()
    mock_get_team_id.assert_called_once()


def test_get_team_id_by_name_returns_expected_id_when_name_exists():
    teams_and_ids = [
        {"name": "Colts", "team_id": 12},
        {"name": "Random", "team_id": 14},
    ]
    assert team_game_logs_service.get_team_id_by_name("Colts", teams_and_ids) == 12


def test_get_team_id_by_name_returns_none_when_name_doesnt_exist():
    teams_and_ids = [
        {"name": "Colts", "team_id": 12},
        {"name": "Random", "team_id": 14},
    ]
    assert team_game_logs_service.get_team_id_by_name("Fake", teams_and_ids) == None


@patch(
    "service.team_game_logs_service.apply_weights",
    side_effect=lambda team, keys, team_id: {**team, "team_id": team_id},
)
def test_normalize_metrics_and_apply_weights(mock_apply_weights):
    team_metrics = [
        {
            "team_id": 1,
            "points_for": 50,
            "points_against": 30,
            "rush_yards_for": 200,
            "rush_yards_against": 150,
            "pass_yards_for": 300,
            "pass_yards_against": 250,
        },
        {
            "team_id": 2,
            "points_for": 70,
            "points_against": 20,
            "rush_yards_for": 180,
            "rush_yards_against": 120,
            "pass_yards_for": 320,
            "pass_yards_against": 280,
        },
    ]

    keys = [
        "points_for",
        "points_against",
        "rush_yards_for",
        "rush_yards_against",
        "pass_yards_for",
        "pass_yards_against",
    ]

    result = team_game_logs_service.normalize_metrics_and_apply_weights(team_metrics)

    expected_metrics = [
        {
            "team_id": 1,
            "points_for": 0.0,  # (50 - 50) / (70 - 50)
            "points_against": 1.0,  # (30 - 20) / (30 - 20)
            "rush_yards_for": 1.0,  # (200 - 180) / (200 - 180)
            "rush_yards_against": 1.0,  # (150 - 120) / (150 - 120)
            "pass_yards_for": 0.0,  # (300 - 300) / (320 - 300)
            "pass_yards_against": 0.0,  # (250 - 250) / (280 - 250)
        },
        {
            "team_id": 2,
            "points_for": 1.0,  # (70 - 50) / (70 - 50)
            "points_against": 0.0,  # (20 - 20) / (30 - 20)
            "rush_yards_for": 0.0,  # (180 - 180) / (200 - 180)
            "rush_yards_against": 0.0,  # (120 - 120) / (150 - 120)
            "pass_yards_for": 1.0,  # (320 - 300) / (320 - 300)
            "pass_yards_against": 1.0,  # (280 - 250) / (280 - 250)
        },
    ]

    for i, team in enumerate(result):
        for key in keys:
            assert (
                team[key] == expected_metrics[i][key]
            ), f"Mismatch for {key} in team {team['team_id']}"


@patch("service.team_game_logs_service.props.get_config", return_value=0.5)
def test_apply_weights_returns_expected_dict(mock_get_config):
    metrics = {
        "points_for": 0.5,
        "points_against": 0.5,
        "rush_yards_for": 0.5,
        "rush_yards_against": 0.5,
        "pass_yards_for": 0.5,
        "pass_yards_against": 0.5,
    }
    keys = [
        "points_for",
        "points_against",
        "rush_yards_for",
        "rush_yards_against",
        "pass_yards_for",
        "pass_yards_against",
    ]

    actual_metrics = team_game_logs_service.apply_weights(metrics, keys, 1)

    expected_metrics = {
        "team_id": 1,
        "points_for": 0.25,
        "points_against": 0.25,
        "rush_yards_for": 0.25,
        "rush_yards_against": 0.25,
        "pass_yards_for": 0.25,
        "pass_yards_against": 0.25,
    }

    assert actual_metrics == expected_metrics


@patch("service.team_game_logs_service.normalize_metrics_and_apply_weights")
def test_calculate_rankings_returns_expected_rankings(mock_normalize_metrics):
    """
    Best off_rush & off_pass should be team 2, best defense should be team 1
    """
    weighted_metrics = [
        {
            "team_id": 1,
            "points_for": 0.5,
            "points_against": 0.5,
            "rush_yards_for": 0.5,
            "rush_yards_against": 0.5,
            "pass_yards_for": 0.5,
            "pass_yards_against": 0.5,
        },
        {
            "team_id": 2,
            "points_for": 0.75,
            "points_against": 0.75,
            "rush_yards_for": 0.75,
            "rush_yards_against": 0.75,
            "pass_yards_for": 0.75,
            "pass_yards_against": 0.75,
        },
    ]

    mock_normalize_metrics.return_value = weighted_metrics

    off_rush_ranks, off_pass_ranks, def_rush_ranks, def_pass_ranks = (
        team_game_logs_service.calculate_rankings(weighted_metrics)
    )

    assert off_rush_ranks == [{"team_id": 2, "rank": 1}, {"team_id": 1, "rank": 2}]
    assert off_pass_ranks == [{"team_id": 2, "rank": 1}, {"team_id": 1, "rank": 2}]
    assert def_rush_ranks == [{"team_id": 1, "rank": 1}, {"team_id": 2, "rank": 2}]
    assert def_pass_ranks == [{"team_id": 1, "rank": 1}, {"team_id": 2, "rank": 2}]


@patch("service.team_game_logs_service.update_teams_rankings")
@patch("service.team_game_logs_service.calculate_rankings")
@patch("service.team_game_logs_service.get_teams_game_logs_for_season")
@patch("service.team_game_logs_service.get_aggregate_season_metrics")
@patch("service.team_game_logs_service.team_service.get_all_teams")
def test_calculate_all_team_rankings_calls_expected_functions(
    mock_get_all_teams,
    mock_get_aggregate_season_metrics,
    mock_get_team_game_logs,
    mock_calculate_rankings,
    mock_update_team_rankings,
):
    mock_list_of_dict = [{"team_id": "1"}]

    mock_get_all_teams.return_value = mock_list_of_dict
    mock_get_aggregate_season_metrics.return_value = mock_list_of_dict
    mock_get_team_game_logs.return_value = mock_list_of_dict
    mock_calculate_rankings.return_value = (
        [{"team_id": 1}],
        [{"team_id": 1}],
        [{"team_id": 1}],
        [{"team_id": 1}],
    )

    team_game_logs_service.calculate_all_teams_rankings(2024)

    mock_get_all_teams.assert_called_once()
    mock_get_team_game_logs.assert_called_once()
    mock_calculate_rankings.assert_called_once()
    mock_get_aggregate_season_metrics.assert_called_once()
    mock_update_team_rankings.assert_called_once()
