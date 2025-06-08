from datetime import datetime
from bs4 import BeautifulSoup
from unittest.mock import patch
import scrape_and_score.scraping.espn as espn


@patch("scrape_and_score.scraping.espn.extract_team_name")
@patch("scrape_and_score.scraping.espn.fetch_page")
@patch("scrape_and_score.scraping.espn.generate_and_persist")
@patch("scrape_and_score.scraping.espn.props.get_config")
def test_scrape_upcoming_games_happy(
    mock_get_config, mock_generate_and_persist, mock_fetch_page, mock_extract_team_name
):
    sample_html = """
    <html>
      <body>
        <div class="Table__Title">August 25, 2024</div>
        <table>
          <tbody class="Table__TBODY">
            <tr>
              <td class="events__col Table__TD"></td>
            </tr>
            <tr>
              <td class="events__col Table__TD"></td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """

    mock_get_config.return_value = (
        "https://www.espn.com/nfl/schedule/_/week/{}/year/{}/seasontype/2"
    )
    mock_fetch_page.return_value = sample_html
    mock_extract_team_name.side_effect = [
        "miami dolphins",
        "buffalo bills",
        "new york jets",
        "new england patriots",
    ]

    espn.scrape_upcoming_games(season=2024, week=1)

    mock_generate_and_persist.assert_called_once()
    records = mock_generate_and_persist.call_args[0][0]

    actual_teams = {(r["home_team"], r["away_team"]) for r in records}
    expected_teams = {
        ("buffalo bills", "miami dolphins"),
        ("new england patriots", "new york jets"),
    }

    assert actual_teams == expected_teams


def test_extract_team_name_away_home():
    html = """
    <td>
        <div>
            <span>
                <a href="/team/_/name/abc">Ignore Me</a>
                <a href="/team/_/name/new-england-patriots">New England Patriots</a>
            </span>
            <span>
                <a href="/team/_/name/xyz">Ignore Me Too</a>
                <a href="/team/_/name/miami-dolphins">Miami Dolphins</a>
            </span>
        </div>
    </td>
    """
    td = BeautifulSoup(html, "html.parser").find("td")

    assert espn.extract_team_name(td, is_home=False) == "patriots"
    assert espn.extract_team_name(td, is_home=True) == "dolphins"


@patch("scrape_and_score.scraping.espn.fetch_game_date_from_team_game_log")
def test_calculate_rest_days_various(mock_prev_date):
    current = datetime(2024, 9, 1)
    mock_prev_date.return_value = datetime(2024, 8, 25)
    assert espn.calculate_rest_days(1, 2024, 2, current) == 7

    mock_prev_date.side_effect = [None, datetime(2023, 12, 25)]
    assert (
        espn.calculate_rest_days(1, 2024, 1, current)
        == (current - datetime(2023, 12, 25)).days
    )

    mock_prev_date.side_effect = [None, None]
    assert espn.calculate_rest_days(1, 2024, 1, current) == 100


@patch("scrape_and_score.scraping.espn.fetch_team_game_log_by_pk")
@patch("scrape_and_score.scraping.espn.update_team_game_log_game_date")
def test_filter_records(mock_update, mock_fetch_pk):
    rec = {"team_id": 1, "game_date": "09/01/2024", "week": 1, "year": 2024}

    # new record
    mock_fetch_pk.return_value = None
    out = espn.filter([rec], 2024, 1)
    assert len(out) == 1

    # same date persisted
    mock_fetch_pk.return_value = {"game_date": datetime(2024, 9, 1).date()}
    out = espn.filter([rec], 2024, 1)
    assert out == []
    mock_update.assert_not_called()

    # date changed from persisted value
    mock_fetch_pk.return_value = {"game_date": datetime(2024, 8, 25).date()}
    out = espn.filter([rec], 2024, 1)
    assert out == []
    mock_update.assert_called_once()


@patch("scrape_and_score.scraping.espn.fetch_all_teams")
def test_generate_team_id_mapping(mock_fetch_all):
    mock_fetch_all.return_value = [
        {"name": "New England Patriots", "team_id": 1},
        {"name": "Chicago Bears", "team_id": 2},
    ]
    mapping = espn.generate_team_id_mapping()
    assert mapping["patriots"] == 1
    assert mapping["bears"] == 2
