from unittest.mock import patch, MagicMock
from scrape_and_score.scraping import our_lads
from bs4 import BeautifulSoup


@patch("scrape_and_score.scraping.our_lads.insert_player_depth_chart_position_records")
@patch("scrape_and_score.scraping.our_lads.insert_player_teams_records")
@patch("scrape_and_score.scraping.our_lads.insert_player_records")
@patch("scrape_and_score.scraping.our_lads.fetch_player_by_normalized_name")
@patch("scrape_and_score.scraping.our_lads.player_service.normalize_name")
@patch("scrape_and_score.scraping.our_lads.scraping_util.fetch_page")
@patch("scrape_and_score.scraping.our_lads.BeautifulSoup")
@patch("scrape_and_score.scraping.our_lads.create_date_week_mapping")
@patch("scrape_and_score.scraping.our_lads.team_service.get_team_id_by_name")
def test_generate_player_and_player_teams_records(
    mock_get_team_id,
    mock_date_map,
    mock_bs,
    mock_fetch_page,
    mock_normalize,
    mock_fetch_player,
    mock_insert_players,
    mock_insert_teams,
    mock_insert_depth,
):
    # arrange
    mock_get_team_id.return_value = 1
    mock_date_map.return_value = {
        "09/01/2023": {"strt_wk": 1, "end_wk": 4}
    }
    mock_fetch_page.return_value = "<html></html>"
    
    soup = MagicMock()
    mock_bs.return_value = soup
    soup.find.return_value.find.return_value = [
        MagicMock(
            find_all=lambda _: [
                MagicMock(get_text=lambda: "WR"),
                MagicMock(get_text=lambda: "Smith, John"),
            ]
        )
    ]

    mock_normalize.return_value = "john-smith"
    mock_fetch_player.return_value = {"player_id": 101}

    archive_dates = [{"season": 2023, "archives": {"09/01/2023": "123"}}]
    teams = [{"team": "Colts", "acronym": "IND"}]

    # act
    our_lads.generate_player_and_player_teams_records(teams, 2023, 2023, archive_dates)

    # assert
    assert mock_get_team_id.called
    assert mock_fetch_page.called
    assert mock_insert_players.called
    assert mock_insert_teams.called
    assert mock_insert_depth.called


@patch("scrape_and_score.scraping.our_lads.update_player_teams_records_end_dates")
@patch("scrape_and_score.scraping.our_lads.insert_player_teams")
@patch("scrape_and_score.scraping.our_lads.fetch_player_teams_records_by_player_and_season")
def test_upsert_player_teams_records_insert_and_update(
    mock_fetch_records, mock_insert, mock_update
):
    # arrange
    mock_fetch_records.return_value = [
        {"team_id": 1, "strt_wk": 1, "end_wk": 18}
    ]
    relevant_players = ["John Smith"]
    mapping = {"John Smith": 42}
    team_id = 2  # different team, triggers update
    season = 2024
    week = 10
    team = {"name": "Colts"}

    # act
    our_lads.upsert_player_teams_records(
        relevant_players, team_id, season, team, mapping, week
    )

    # assert
    assert mock_update.called
    assert mock_insert.called


@patch("scrape_and_score.scraping.our_lads.insert_players")
@patch("scrape_and_score.scraping.our_lads.is_previously_inserted_player")
def test_insert_player_records_skips_existing(mock_is_inserted, mock_insert):

    # arrange
    mock_is_inserted.return_value = True
    player_records = [{"name": "John Smith"}]
    team = {"team": "Colts"}
    mapping = {}

    # act
    our_lads.insert_player_records(player_records, team, mapping, 2024)

    # assert
    mock_insert.assert_not_called()


def test_start_and_end_date_players_on_team():

    # arrange
    relevant_players = set(["Jane Doe"])
    current_players = [{"name": "John Smith"}]  # jane removed, john added
    start_mapping = {"Jane Doe": 1}
    end_mapping = {}
    date_map = {
        "09/01/2023": {"strt_wk": 1, "end_wk": 4},
        "10/01/2023": {"strt_wk": 5, "end_wk": 8}
    }

    # act
    our_lads.start_and_end_date_players_on_team(
        relevant_players,
        current_players,
        start_mapping,
        end_mapping,
        date_map,
        "10/01/2023",
        "09/01/2023",
        2023,
    )

    # assert
    assert "Jane Doe" in end_mapping
    assert "John Smith" in start_mapping



@patch("scrape_and_score.scraping.our_lads.generate_player_and_player_teams_records")
@patch("scrape_and_score.scraping.our_lads.extract_archive_dates")
@patch("scrape_and_score.scraping.our_lads.BeautifulSoup")
@patch("scrape_and_score.scraping.our_lads.scraping_util.fetch_page")
@patch("scrape_and_score.scraping.our_lads.props.get_config")
def test_scrape_and_persist(mock_get_config, mock_fetch_page, mock_bs, mock_extract_dates, mock_generate_records):
    
    # arrange
    mock_get_config.return_value = [{"name": "Colts", "pfr_acronym": "IND"}]
    mock_fetch_page.return_value = "<html></html>"
    mock_bs.return_value.find_all.return_value = ["option1", "option2"]
    mock_extract_dates.return_value = [{"season": 2023, "archives": {"09/01/2023": "150"}}]

    # act
    our_lads.scrape_and_persist(2023, 2023)

    # assert
    assert mock_get_config.called
    assert mock_fetch_page.called
    assert mock_extract_dates.called
    assert mock_generate_records.called


@patch("scrape_and_score.scraping.our_lads.generate_and_persist_depth_chart_records")
@patch("scrape_and_score.scraping.our_lads.props.get_config")
def test_scrape_and_persist_upcoming(mock_get_config, mock_generate):
    # arrange
    mock_get_config.return_value = [{"name": "Colts", "pfr_acronym": "IND"}]

    # act
    our_lads.scrape_and_persist_upcoming(2024, 1, is_update=False)

    # assert
    assert mock_get_config.called
    assert mock_generate.called
    args, _ = mock_generate.call_args

    # validate call args
    assert args[1] == 2024  
    assert args[2] == 1     
    assert args[3] is False 



def test_create_date_week_mapping():
    mapping = our_lads.create_date_week_mapping(2024)
    assert mapping["09/01/2024"]["strt_wk"] == 1
    assert mapping["01/01/2025"]["end_wk"] == 18


def test_parse_name_regular():
    result = our_lads.parse_name("Smith, John")
    assert result == "John Smith"


def test_parse_name_ir():
    result = our_lads.parse_name("Doe, Jane WR", is_injured_reserve=True)
    assert result == ("Jane Doe", "WR")


def test_extract_fantasy_relevant_players():
    html = """
    <table><tbody>
        <tr><td>QB</td><td>Smith, John</td></tr>
        <tr><td>WR</td><td>Doe, Jane</td></tr>
        <tr><td>RB</td><td>Williams, Mike</td></tr>
        <tr><td>IR</td><td>Brown, Tom RB</td></tr>
        <tr><td>FB</td><td>Ignore, Me</td></tr>
    </tbody></table>
    """
    soup = BeautifulSoup(html, "html.parser")
    depth_chart = soup.find("tbody").find_all("tr")
    players = our_lads.extract_fantasy_relevant_players(depth_chart)

    assert len(players) == 4
    assert players[0]["name"] == "John Smith"
    assert players[-1]["depth_chart_pos"] == -1  # ir player


@patch("scrape_and_score.scraping.our_lads.player_service.normalize_name")
@patch("scrape_and_score.scraping.our_lads.fetch_player_by_normalized_name")
def test_is_previously_inserted_player_true(mock_fetch, mock_normalize):
    mock_normalize.return_value = "john-smith"
    mock_fetch.return_value = {"player_id": 101}
    player_name_id_mapping = {}
    result = our_lads.is_previously_inserted_player("John Smith", player_name_id_mapping)

    assert result is True
    assert player_name_id_mapping["John Smith"] == 101


@patch("scrape_and_score.scraping.our_lads.player_service.normalize_name")
@patch("scrape_and_score.scraping.our_lads.fetch_player_by_normalized_name")
def test_is_previously_inserted_player_false(mock_fetch, mock_normalize):
    mock_normalize.return_value = "unknown"
    mock_fetch.return_value = None
    result = our_lads.is_previously_inserted_player("Ghost", {})
    assert result is False


def test_create_date_week_mapping_keys():
    mapping = our_lads.create_date_week_mapping(2022)
    expected_keys = [
        "09/01/2022",
        "10/01/2022",
        "11/01/2022",
        "12/01/2022",
        "01/01/2023"
    ]
    assert list(mapping.keys()) == expected_keys
