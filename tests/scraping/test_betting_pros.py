import pytest
from unittest.mock import patch, MagicMock
from scrape_and_score.scraping import betting_pros
import requests


@pytest.fixture
def sample_offer_data():
    return {
        "offers": [
            {
                "market_id": "1",
                "selections": [
                    {
                        "label": "Over",
                        "books": [
                            {
                                "lines": [
                                    {"best": True, "cost": -115, "line": 74.5},
                                    {"best": False, "cost": -120, "line": 75.5},
                                ]
                            }
                        ],
                    },
                    {
                        "label": "Under",
                        "books": [
                            {"lines": [{"best": True, "cost": -110, "line": 74.5}]}
                        ],
                    },
                ],
            }
        ],
        "_pagination": {"total_pages": 1},
    }


@patch("scrape_and_score.scraping.betting_pros.insert_player_props")
@patch("scrape_and_score.scraping.betting_pros.get_player_betting_odds")
@patch("scrape_and_score.scraping.betting_pros.fetch_event_ids_for_week")
@patch("scrape_and_score.scraping.betting_pros.fetch_players_active_in_specified_year")
@patch("scrape_and_score.scraping.betting_pros.props.get_config")
@patch(
    "scrape_and_score.scraping.betting_pros.fetch_max_week_persisted_in_team_betting_odds_table"
)
def test_fetch_historical_odds_happy_path(
    mock_max_week,
    mock_get_config,
    mock_fetch_players,
    mock_fetch_events,
    mock_get_odds,
    mock_insert_props,
):
    mock_max_week.return_value = 2
    mock_get_config.return_value = ["market1"]
    mock_fetch_players.return_value = [
        {"id": 1, "name": "Tom Brady"},
        {"id": 2, "name": "T.J. O'Neill"},
    ]
    mock_fetch_events.return_value = [1001, 1002]
    mock_get_odds.return_value = {"passing_yards": 275.5}

    betting_pros.fetch_historical_odds(2024)

    assert mock_insert_props.call_count == 2
    assert mock_get_odds.call_count == 4  # 2 players * 2 weeks
    mock_get_odds.assert_any_call("tom-brady", [1001, 1002], ["market1"])
    mock_get_odds.assert_any_call("tj-oneill", [1001, 1002], ["market1"])


@patch("scrape_and_score.scraping.betting_pros.insert_player_props")
@patch(
    "scrape_and_score.scraping.betting_pros.get_player_betting_odds", return_value=None
)
@patch(
    "scrape_and_score.scraping.betting_pros.fetch_event_ids_for_week", return_value=[1]
)
@patch("scrape_and_score.scraping.betting_pros.fetch_players_active_in_specified_year")
@patch("scrape_and_score.scraping.betting_pros.props.get_config", return_value=["mkt"])
@patch(
    "scrape_and_score.scraping.betting_pros.fetch_max_week_persisted_in_team_betting_odds_table",
    return_value=1,
)
def test_fetch_historical_odds_no_odds(
    _, __, mock_fetch_players, ___, ____, mock_insert
):
    mock_fetch_players.return_value = [{"id": 1, "name": "Tom Brady"}]

    betting_pros.fetch_historical_odds(2024)

    mock_insert.assert_not_called()


@patch("scrape_and_score.scraping.betting_pros.fetch_upcoming_game_conditions")
@patch("scrape_and_score.scraping.betting_pros.fetch_upcoming_player_odds")
def test_fetch_upcoming_player_odds_and_game_conditions(mock_odds, mock_conditions):
    betting_pros.fetch_upcoming_player_odds_and_game_conditions(2, 2024, [1, 2, 3])
    mock_odds.assert_called_once_with(2, 2024, [1, 2, 3])
    mock_conditions.assert_called_once_with(2, 2024)


@patch("scrape_and_score.scraping.betting_pros.update_game_conditions")
@patch("scrape_and_score.scraping.betting_pros.insert_game_conditions")
@patch("scrape_and_score.scraping.betting_pros.filter_game_conditions")
@patch("scrape_and_score.scraping.betting_pros.rotowire.create_team_id_mapping")
@patch("scrape_and_score.scraping.betting_pros.get_data")
@patch("scrape_and_score.scraping.betting_pros.props.get_config")
def test_fetch_upcoming_game_conditions_happy(
    mock_config,
    mock_get_data,
    mock_mapping,
    mock_filter,
    mock_insert,
    mock_update,
):
    mock_config.return_value = "http://mock.url/{WEEK}/{YEAR}"
    mock_get_data.return_value = {
        "events": [
            {
                "venue": {},
                "weather": {
                    "forecast_temp": 60,
                    "forecast_wind_degree": 180,
                    "forecast_wind_speed": 12,
                    "forecast_rain_chance": 0.2,
                    "forecast_icon": "rain",
                },
                "scheduled": "2024-09-01 13:00:00",
                "home": "NE",
                "visitor": "MIA",
            }
        ]
    }
    mock_mapping.return_value = {"NE": 1, "MIA": 2}
    mock_filter.return_value = ([{"mock": "update"}], [{"mock": "insert"}])

    betting_pros.fetch_upcoming_game_conditions(1, 2024)

    mock_insert.assert_called_once()
    mock_update.assert_called_once()


@patch("scrape_and_score.scraping.betting_pros.are_game_conditions_modified")
@patch("scrape_and_score.scraping.betting_pros.fetch_game_conditions_record_by_pk")
def test_filter_game_conditions(mock_fetch_pk, mock_modified):
    mock_fetch_pk.side_effect = [None, {"existing": True}, {"existing": True}]
    mock_modified.side_effect = [True, False]

    records = [
        {"season": 2024, "week": 1, "home_team_id": 1, "visit_team_id": 2},
        {"season": 2024, "week": 1, "home_team_id": 3, "visit_team_id": 4},
        {"season": 2024, "week": 1, "home_team_id": 5, "visit_team_id": 6},
    ]

    update, insert = betting_pros.filter_game_conditions(records)

    assert len(insert) == 1
    assert len(update) == 1


def test_are_game_conditions_modified_no_change():
    record = {
        "game_date": "2025-09-10",
        "game_time": 20,
        "kickoff": "Sep 10 8:00 PM",
        "month": "September",
        "start": "Night",
        "surface": "Turf",
        "weather_icon": "partly-cloudy-night",
        "temperature": 70.0,
        "precip_probability": 10,
        "precip_type": "Rain",
        "wind_speed": 5,
        "wind_bearing": 180,
    }
    assert betting_pros.are_game_conditions_modified(record, record) is False


@pytest.mark.parametrize(
    "icon,expected",
    [
        ("light-rain-day", "Rain"),
        ("heavy-snow", "Snow"),
        ("partly-cloudy-night", None),
        (None, None),
    ],
)
def test_extract_precip_type(icon, expected):
    assert betting_pros.extract_precip_type(icon) == expected


def test_extract_game_time_metrics_day():
    game_date, game_time, kickoff, month, start = (
        betting_pros.extract_game_time_metrics("2025-09-10 08:30:00")
    )
    assert game_time == 8
    assert start == "Day"
    assert month == "September"


def test_are_game_conditions_modified_with_change():
    persisted = {
        "temperature": 70.0,
    }
    current = {
        "temperature": 65.0,
    }
    assert betting_pros.are_game_conditions_modified(persisted, current) is True


def test_extract_surface_dome():
    venue = {"stadium_type": "retractable_dome", "surface": "turf"}
    assert betting_pros.extract_surface(venue) == "Dome"


def test_extract_surface_turf():
    venue = {"stadium_type": "open", "surface": "artificial"}
    assert betting_pros.extract_surface(venue) == "Turf"


def test_extract_surface_grass():
    venue = {"stadium_type": "open", "surface": "natural grass"}
    assert betting_pros.extract_surface(venue) == "Grass"


def test_extract_surface_none():
    assert betting_pros.extract_surface({}) == ""


@patch("scrape_and_score.scraping.betting_pros.fetch_player_betting_odds_record_by_pk")
def test_are_odds_modified(mock_fetch):
    persisted = {"cost": 120, "line": 50.5}
    current_same = {"cost": 120, "line": 50.5}
    current_modified = {"cost": 125, "line": 50.5}

    assert betting_pros.are_odds_modified(persisted, current_same) is False
    assert betting_pros.are_odds_modified(persisted, current_modified) is True


@patch("scrape_and_score.scraping.betting_pros.fetch_player_betting_odds_record_by_pk")
def test_filter_upcoming_player_odds(mock_fetch):
    mock_fetch.side_effect = [None, {"cost": 110, "line": 45.5}]
    records = [
        {
            "player_id": 1,
            "player_name": "John Doe",
            "week": 3,
            "season": 2025,
            "odds": [
                {"label": "Receiving Yards", "cost": 110, "line": 55.5},
                {"label": "Rushing Yards", "cost": 115, "line": 45.5},
            ],
        }
    ]

    update_records, insert_records = betting_pros.filter_upcoming_player_odds(records)

    assert len(insert_records) == 1
    assert len(update_records) == 1
    assert insert_records[0]["label"] == "Receiving Yards"
    assert update_records[0]["label"] == "Rushing Yards"


@patch(
    "scrape_and_score.scraping.betting_pros.props.get_config", return_value=":101:102"
)
@patch(
    "scrape_and_score.scraping.betting_pros.fetch_event_ids_for_week",
    return_value=[1, 2],
)
@patch("scrape_and_score.scraping.betting_pros.get_player_betting_odds")
@patch("scrape_and_score.scraping.betting_pros.player_service.get_player_name_by_id")
@patch("scrape_and_score.scraping.betting_pros.filter_upcoming_player_odds")
@patch("scrape_and_score.scraping.betting_pros.insert_upcoming_player_props")
@patch("scrape_and_score.scraping.betting_pros.update_upcoming_player_props")
def test_fetch_upcoming_player_odds(
    mock_update,
    mock_insert,
    mock_filter,
    mock_get_name,
    mock_get_odds,
    mock_fetch_event_ids,
    mock_get_config,
):
    mock_get_name.return_value = "John Doe"
    mock_get_odds.return_value = [{"label": "Total Yards", "line": 80.5, "cost": 120}]
    mock_filter.return_value = ([], [{"dummy": "record"}])  # only insert

    betting_pros.fetch_upcoming_player_odds(week=2, season=2025, player_ids=[42])

    mock_fetch_event_ids.assert_called_once_with(2, 2025)
    mock_get_name.assert_called_once_with(42)
    mock_get_odds.assert_called_once()
    mock_filter.assert_called_once()
    mock_insert.assert_called_once()
    mock_update.assert_not_called()


@patch(
    "scrape_and_score.scraping.betting_pros.props.get_config",
    return_value="https://dummy.com/{MARKET_IDS}/{PLAYER_SLUG}/{EVENT_IDS}/{PAGE}",
)
@patch("scrape_and_score.scraping.betting_pros.get_data")
def test_get_player_betting_odds_single_page(
    mock_get_data, mock_get_config, sample_offer_data
):
    mock_get_data.return_value = sample_offer_data
    betting_pros.MARKET_ID_MAPPING = {"1": "Passing Yards"}

    odds = betting_pros.get_player_betting_odds("patrick-mahomes", "123", "1")

    assert len(odds) == 1
    assert odds[0]["label"] == "Passing Yards (Over)"
    assert odds[0]["cost"] == -115
    assert odds[0]["line"] == 74.5


def test_determine_number_of_pages_with_pagination(sample_offer_data):
    pages = betting_pros.determine_number_of_pages(sample_offer_data)
    assert pages == 1


def test_determine_number_of_pages_no_pagination():
    assert betting_pros.determine_number_of_pages({}) == 0


def test_get_odds_filters_and_selects_best(sample_offer_data):
    market_ids = {"1": "Passing Yards"}
    odds = betting_pros.get_odds(sample_offer_data, market_ids)

    assert len(odds) == 1
    assert odds[0]["label"] == "Passing Yards (Over)"
    assert odds[0]["cost"] == -115
    assert odds[0]["line"] == 74.5


@patch("scrape_and_score.scraping.betting_pros.requests.get")
@patch(
    "scrape_and_score.scraping.betting_pros.props.get_config",
    side_effect=["0", "dummy-api-key"],
)
@patch("scrape_and_score.scraping.betting_pros.time.sleep", return_value=None)
def test_get_data_returns_json(mock_sleep, mock_get_config, mock_requests_get):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"key": "value"}
    mock_requests_get.return_value = mock_response

    result = betting_pros.get_data("https://example.com")
    assert result == {"key": "value"}


@patch(
    "scrape_and_score.scraping.betting_pros.requests.get",
    side_effect=requests.RequestException("Network error"),
)
@patch(
    "scrape_and_score.scraping.betting_pros.props.get_config",
    return_value="dummy-api-key",
)
@patch("scrape_and_score.scraping.betting_pros.time.sleep", return_value=None)
def test_get_data_network_error(mock_sleep, mock_get_config, mock_requests_get):
    result = betting_pros.get_data("https://example.com")
    assert result is None


@patch(
    "scrape_and_score.scraping.betting_pros.get_data",
    return_value={"events": [{"id": 1}, {"id": 2}]},
)
@patch(
    "scrape_and_score.scraping.betting_pros.props.get_config",
    return_value="https://dummy.com/{WEEK}/{YEAR}",
)
def test_fetch_event_ids_for_week(mock_get_config, mock_get_data):
    result = betting_pros.fetch_event_ids_for_week(3, 2024)
    assert result == "1:2"
