from service import team_service
from unittest.mock import patch


@patch("service.team_service.fetch_data.fetch_team_by_name")
def test_get_team_id_by_name_returns_expected_id_when_name_persisted(
    mock_fetch_team_by_name,
):
    expected_team_id = 12
    mock_fetch_team_by_name.return_value = {"team_id": expected_team_id}
    assert team_service.get_team_id_by_name("Apple") == expected_team_id


@patch("service.team_service.fetch_data.fetch_team_by_name")
def test_get_team_id_by_name_returns_none_when_name_not_persisted(
    mock_fetch_team_by_name,
):
    mock_fetch_team_by_name.return_value = None
    assert team_service.get_team_id_by_name("Apple") == None


@patch("service.team_service.fetch_data.fetch_all_teams")
def test_get_all_teams_when_none_persisted(mock_fetch_all_teams):
    mock_fetch_all_teams.return_value = []
    assert team_service.get_all_teams() == []


@patch("service.team_service.fetch_data.fetch_all_teams")
def test_get_all_teams_when_teams_are_persisted(mock_fetch_all_teams):
    expected_teams = [{"Team A"}, {"Team B"}]
    mock_fetch_all_teams.return_value = expected_teams
    assert team_service.get_all_teams() == expected_teams
