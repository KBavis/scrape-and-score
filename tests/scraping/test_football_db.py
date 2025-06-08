import pytest
from unittest.mock import patch
from bs4 import BeautifulSoup
import scrape_and_score.scraping.football_db as football_db


@pytest.fixture
def sample_player_html():
    return """
    <div class="tr">
        <a>John Doe</a>
        <div class="td w15 d-none d-md-table-cell">knee</div>
        <div class="td center w15 d-none d-md-table-cell">DNP</div>
        <div class="td center w15 d-none d-md-table-cell">LP</div>
        <div class="td center w15 d-none d-md-table-cell">FP</div>
        <div class="td w20 d-none d-md-table-cell">(12/14) <b>Out</b> @ KC</div>
    </div>
    """


@pytest.fixture
def sample_soup(sample_player_html):
    html = f"""
    <div class="divtable divtable-striped divtable-mobile">
        {sample_player_html}
    </div>
    """
    return BeautifulSoup(html, "html.parser")


def test_extract_player_statuses_and_injury(sample_soup):
    player_div = sample_soup.find("div", class_="tr")
    result = football_db.extract_player_statuses_and_injury(player_div)

    assert result["player_name"] == "John Doe"
    assert result["injury_locations"] == "knee"
    assert result["wed_prac_sts"].lower() == "dnp"
    assert result["thurs_prac_sts"].lower() == "lp"
    assert result["fri_prac_sts"].lower() == "fp"
    assert result["off_sts"].lower() == "out"


def test_parse_team_injuries(sample_soup):
    injuries = football_db.parse_team_injuries(sample_soup)
    assert len(injuries) == 1
    assert injuries[0]["player_name"] == "John Doe"


def test_parse_all_player_injuries(sample_soup):
    result = football_db.parse_all_player_injuries(sample_soup)
    assert isinstance(result, list)
    assert len(result) == 1


def test_normalize_status():
    assert football_db.normalize_status("DNP") == "dnp"
    assert football_db.normalize_status("--") is None


def test_extract_game_status():
    # Updated test inputs to match expected format with bold tags
    assert (
        football_db.extract_game_status("(12/14) <b>Questionable</b> @ KC")
        == "questionable"
    )
    assert football_db.extract_game_status("--") is None
    assert football_db.extract_game_status("random text") is None


@patch(
    "scrape_and_score.scraping.football_db.fetch_pks_for_inserted_player_injury_records"
)
@patch("scrape_and_score.scraping.football_db.insert_player_injuries")
@patch("scrape_and_score.scraping.football_db.update_player_injuries")
@patch(
    "scrape_and_score.scraping.football_db.player_service.get_player_id_by_normalized_name"
)
@patch("scrape_and_score.scraping.football_db.player_service.normalize_name")
def test_generate_and_persist_player_injury_records(
    mock_normalize, mock_get_id, mock_update, mock_insert, mock_fetch_pks
):
    mock_fetch_pks.return_value = [{"player_id": 1, "week": 1, "season": 2023}]
    mock_get_id.return_value = 1
    mock_normalize.return_value = "john doe"

    record = {
        "player_name": "John Doe",
        "injury_locations": "knee",
        "wed_prac_sts": "dnp",
        "thurs_prac_sts": "lp",
        "fri_prac_sts": "fp",
        "off_sts": "out",
    }

    football_db.generate_and_persist_player_injury_records(
        [record], season=2023, week=1, player_ids=[1]
    )

    mock_update.assert_called_once()
    mock_insert.assert_not_called()


def test_filter_persisted_records():
    new_records = [
        {"player_id": 1, "week": 1, "season": 2023},
        {"player_id": 2, "week": 1, "season": 2023},
    ]
    persisted = [{"player_id": 1, "week": 1, "season": 2023}]

    updates, inserts = football_db.filter_persisted_records(new_records, persisted)
    assert len(updates) == 1
    assert len(inserts) == 1


@patch("scrape_and_score.scraping.football_db.props.get_config")
@patch("scrape_and_score.scraping.football_db.scraping_util.fetch_page")
@patch(
    "scrape_and_score.scraping.football_db.generate_and_persist_player_injury_records"
)
def test_scrape_upcoming(mock_persist, mock_fetch_page, mock_get_config, sample_soup):
    mock_fetch_page.return_value = str(sample_soup)
    mock_get_config.return_value = (
        "https://www.footballdb.com/transactions/injuries.html?yr={}&wk={}&type=reg"
    )
    football_db.scrape_upcoming(week=1, season=2023, player_ids=[1])
    assert mock_fetch_page.called
    assert mock_persist.called


@patch("scrape_and_score.scraping.football_db.props.get_config")
@patch("scrape_and_score.scraping.football_db.scraping_util.fetch_page")
@patch(
    "scrape_and_score.scraping.football_db.generate_and_persist_player_injury_records"
)
def test_scrape_historical(mock_persist, mock_fetch_page, mock_get_config, sample_soup):
    mock_fetch_page.return_value = str(sample_soup)
    mock_get_config.return_value = (
        "https://www.footballdb.com/transactions/injuries.html?yr={}&wk={}&type=reg"
    )
    football_db.scrape_historical(start_year=2023, end_year=2023)
    assert mock_fetch_page.call_count == 18
    assert mock_persist.call_count == 18
