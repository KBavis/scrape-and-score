import pytest
from bs4 import BeautifulSoup
from scrape_and_score.scraping.pfr import utils
from unittest.mock import patch


def test_get_last_name_first_initial():
    assert utils.get_last_name_first_initial("Tom Brady") == "B"
    with pytest.raises(Exception):
        utils.get_last_name_first_initial("Brady")


def test_check_name_similarity():
    similarity = utils.check_name_similarity("Tom Brady", "Tom Brady")
    assert similarity == 100


def test_order_players_by_last_name():
    players = [
        {"player_name": "Tom Brady"},
        {"player_name": "Aaron Rodgers"},
        {"player_name": "Josh Allen"},
    ]
    result = utils.order_players_by_last_name(players)
    assert len(result["B"]) == 1
    assert len(result["R"]) == 1
    assert len(result["A"]) == 1


def test_calculate_distance():
    city1 = {"latitude": 40.7128, "longitude": -74.0060}  # NYC
    city2 = {"latitude": 34.0522, "longitude": -118.2437}  # LA
    dist = utils.calculate_distance(city1, city2)
    assert 2400 < dist < 3000


def test_parse_team_totals():
    html = '<table><tfoot><tr><td data-stat="xpa">10</td><td data-stat="xpm">8</td></tr></tfoot></table>'
    soup = BeautifulSoup(html, "html.parser")
    tfoot = soup.find("tfoot")
    result = utils.parse_team_totals(tfoot)
    assert result == {"team_total_xpa": "10", "team_total_xpm": "8"}


def test_filter_metrics_by_week():
    metrics = [
        {"week": "1", "yards": 100},
        {"week": "1", "yards": 200},
        {"week": "2", "yards": 150},
    ]
    filtered = utils.filter_metrics_by_week(metrics)
    assert len(filtered) == 2
    assert filtered[0]["week"] == "1"
    assert filtered[1]["week"] == "2"


def test_is_team_game_log_modified():
    current = {"result": "W", "points_for": 21, "points_allowed": 17, "off_tot_yds": 350,
               "off_pass_yds": 250, "off_rush_yds": 100, "def_tot_yds": 300,
               "def_pass_yds": 180, "def_rush_yds": 120, "pass_tds": 2, "pass_cmp": 20,
               "pass_att": 30, "pass_cmp_pct": 66.7, "rush_att": 25, "rush_tds": 1,
               "yds_gained_per_pass_att": 8.3, "adj_yds_gained_per_pass_att": 7.9,
               "pass_rate": 98.3, "sacked": 2, "sack_yds_lost": 12, "rush_yds_per_att": 4.0,
               "total_off_plays": 60, "yds_per_play": 5.8, "fga": 2, "fgm": 1,
               "xpa": 3, "xpm": 3, "total_punts": 4, "punt_yds": 180,
               "pass_fds": 15, "rsh_fds": 5, "pen_fds": 2, "total_fds": 22,
               "thrd_down_conv": 6, "thrd_down_att": 12, "fourth_down_conv": 1,
               "fourth_down_att": 2, "penalties": 4, "penalty_yds": 35,
               "fmbl_lost": 1, "interceptions": 1, "turnovers": 2, "time_of_poss": 29.5}
    persisted = current.copy()
    assert utils.is_team_game_log_modified(current, persisted) is False
    current["points_for"] = 28
    assert utils.is_team_game_log_modified(current, persisted) is True


def test_get_additional_metrics_qb():
    fields = utils.get_additional_metrics("QB")
    assert "cmp" in fields and isinstance(fields["cmp"], list)


def test_get_additional_metrics_invalid():
    with pytest.raises(Exception):
        utils.get_additional_metrics("K")
