import pytest
from unittest.mock import patch
import pandas as pd
import scrape_and_score.scraping.rotowire as rotowire


@pytest.fixture
def sample_json():
    return [
        {
            "season": "2023",
            "week": "1",
            "home_team_stats_id": "DAL",
            "visit_team_stats_id": "NYG",
            "game_date": "2023-09-01",
            "game_time": "1:00 PM",
            "kickoff": "1:00 PM",
            "month": "September",
            "start": "1:00 PM",
            "surface": "Grass",
            "weather_icon": "sunny",
            "temperature": 75,
            "precip_probability": 0.1,
            "precip_type": "none",
            "wind_speed": 5,
            "wind_bearing": "N",
            "home_team_score": 21,
            "visit_team_score": 17,
            "game_over_under": 38,
            "favorite": "DAL",
            "spread": 3,
            "total": 38,
            "over_hit": True,
            "under_hit": False,
            "favorite_covered": True,
            "underdog_covered": False
        }
    ]


@patch("scrape_and_score.scraping.rotowire.insert_teams_odds")
@patch("scrape_and_score.scraping.rotowire.insert_game_conditions")
@patch("scrape_and_score.scraping.rotowire.props.get_config")
@patch("scrape_and_score.scraping.rotowire.requests.get")
@patch("scrape_and_score.scraping.rotowire.create_team_id_mapping")
def test_scrape_all(mock_create_team_id_mapping, mock_get, mock_config, mock_insert_conditions, mock_insert_odds, sample_json):
    # arrange
    mock_config.side_effect = lambda k: "2023" if k == "nfl.current-year" else "https://fake.history"
    mock_get.return_value.json.return_value = sample_json
    mock_create_team_id_mapping.return_value = {"DAL": 10, "NYG": 20}

    # act
    rotowire.scrape_all()

    # assert 
    assert mock_insert_odds.called
    assert mock_insert_conditions.called


def test_are_odds_modified_false():
    persisted = {
        "home_team_id": 1,
        "away_team_id": 2,
        "week": 1,
        "season": 2024,
        "favorite_team_id": 1,
        "game_over_under": 44.0,
        "spread": -3.5
    }
    current = {
        "home_team_id": 1,
        "away_team_id": 2,
        "week": 1,
        "year": 2024,
        "favorite_team_id": 1,
        "game_over_under": 44.0,
        "spread": -3.5
    }
    assert not rotowire.are_odds_modified(persisted, current)


def test_are_odds_modified_true():
    persisted = {
        "home_team_id": 1,
        "away_team_id": 2,
        "week": 1,
        "season": 2024,
        "favorite_team_id": 1,
        "game_over_under": 44.0,
        "spread": -3.5
    }
    current = {
        "home_team_id": 1,
        "away_team_id": 2,
        "week": 1,
        "year": 2024,
        "favorite_team_id": 99,
        "game_over_under": 44.0,
        "spread": -3.5
    }
    assert rotowire.are_odds_modified(persisted, current)


@patch("scrape_and_score.scraping.rotowire.create_team_id_mapping")
def test_get_game_conditions(mock_create_team_id_mapping):
    mock_create_team_id_mapping.return_value = {"DAL": 10, "NYG": 20}
    df = pd.DataFrame([{
        "season": "2024",
        "week": "2",
        "home_team_stats_id": "DAL",
        "visit_team_stats_id": "NYG",
        "game_date": "2024-09-08",
        "game_time": "6:00 PM",
        "kickoff": "18:00",
        "month": "September",
        "start": "Night",
        "surface": "Grass",
        "weather_icon": "cloudy",
        "temperature": 72.0,
        "precip_probability": 0.0,
        "precip_type": "none",
        "wind_speed": 8.0,
        "wind_bearing": "S"
    }])
    result = rotowire.get_game_conditions(df)
    assert len(result) == 1
    rec = result[0]
    assert rec["home_team_id"] == 10
    assert rec["visit_team_id"] == 20
    assert rec["weather_icon"] == "cloudy"


@patch("scrape_and_score.scraping.rotowire.create_team_id_mapping")
def test_get_team_betting_odds_records(mock_create_team_id_mapping):
    mock_create_team_id_mapping.return_value = {"PHI": 5, "DAL": 6}
    df = pd.DataFrame([{
        "season": "2025",
        "week": "3",
        "home_team_stats_id": "PHI",
        "visit_team_stats_id": "DAL",
        "home_team_score": 30,
        "visit_team_score": 28,
        "game_over_under": 55,
        "favorite": "",
        "spread": -2.5,
        "total": 60,
        "over_hit": False,
        "under_hit": True,
        "underdog_covered": True,
        "favorite_covered": False
    }])
    result = rotowire.get_team_betting_odds_records(df)
    assert len(result) == 1
    rec = result[0]
    assert rec["home_team_id"] == 5
    assert rec["away_team_id"] == 6
    assert rec["favorite_team_id"] == 5  # defaults to home team
    assert rec["over_hit"] is False
    assert rec["under_hit"] is True
    assert rec["underdog_covered"] is True


def test_generate_update_records():
    recent_df = pd.DataFrame([{
        "home_team_stats_id": "SF",
        "visit_team_stats_id": "LA",
        "home_team_score": 24,
        "visit_team_score": 20,
        "total": 44,
        "over_hit": True,
        "under_hit": False,
        "favorite_covered": True,
        "underdog_covered": False
    }])
    mapping = {"SF": 100, "LA": 200}
    result = rotowire.generate_update_records(recent_df, mapping, year=2024, week=4)
    assert len(result) == 1
    rec = result[0]
    assert rec["home_team_id"] == 100
    assert rec["visit_team_id"] == 200
    assert rec["year"] == 2024
    assert rec["week"] == 4

