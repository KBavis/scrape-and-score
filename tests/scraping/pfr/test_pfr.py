from unittest.mock import patch
from bs4 import BeautifulSoup
from scrape_and_score.scraping.pfr import pfr


@patch("scrape_and_score.scraping.pfr.pfr.props.get_config")
@patch("scrape_and_score.scraping.pfr.pfr.fetch_team_game_logs")
@patch("scrape_and_score.scraping.pfr.pfr.fetch_player_metrics")
def test_scrape_all(mock_fetch_player_metrics, mock_fetch_team_game_logs, mock_get_config):
    mock_get_config.side_effect = lambda k: "2023" if k == "nfl.current-year" else "https://team-url"
    mock_fetch_team_game_logs.return_value = ["team_data"]
    mock_fetch_player_metrics.return_value = ["player_data"]

    team_and_player_data = [{"name": "Tom Brady", "position": "QB"}]
    teams = ["Buccaneers"]

    result = pfr.scrape_all(team_and_player_data, teams)
    assert result == (["team_data"], ["player_data"])


def test_parse_advanced_passing_table():
    html = '''
    <table>
        <tbody>
            <tr>
                <td data-stat="week_num">1</td>
                <td data-stat="pass_first_down">5</td>
                <td data-stat="pass_drop_pct">10%</td>
                <td data-stat="pass_hits">2</td>
            </tr>
        </tbody>
    </table>
    '''
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    result = pfr.parse_advanced_passing_table(table)
    assert result[0]["week"] == "1"
    assert result[0]["first_downs"] == "5"
    assert result[0]["drop_pct"] == "10%"
    assert result[0]["hits"] == "2"


def test_parse_advanced_rushing_receiving_table():
    html = '''
    <table>
        <tbody>
            <tr>
                <td data-stat="week_num">1</td>
                <td data-stat="rush_first_down">3</td>
                <td data-stat="rush_yac_per_rush">2.5</td>
                <td data-stat="rec_drops">1</td>
            </tr>
        </tbody>
    </table>
    '''
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    result = pfr.parse_advanced_rushing_receiving_table(table)
    assert result[0]["week"] == "1"
    assert result[0]["rush_first_downs"] == "3"
    assert result[0]["rush_yds_after_contact_per_att"] == "2.5"
    assert result[0]["dropped_passes"] == "1"


def test_extract_team_game_log_updates_no_games():
    soup = BeautifulSoup("<html><body></body></html>", "html.parser")
    result = pfr.extract_team_game_log_updates(soup, 1, 2024)
    assert result is None


def test_parse_conversions():
    html = '''
    <tbody>
        <tr><th>Team Stats</th><td data-stat="third_down_pct">40%</td></tr>
        <tr><th>Opp. Stats</th><td data-stat="third_down_pct">35%</td></tr>
    </tbody>
    '''
    soup = BeautifulSoup(html, "html.parser")
    result = pfr.parse_conversions(soup)
    assert result["team_third_down_pct"] == "40%"
    assert result["opp_third_down_pct"] == "35%"


def test_parse_stats():
    html = '''
    <tbody>
        <tr><th>Team Stats</th><td data-stat="total_yds">400</td><td data-stat="points">27</td></tr>
        <tr><th>Opp. Stats</th><td data-stat="total_yds">350</td><td data-stat="points">24</td></tr>
    </tbody>
    '''
    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody")
    result = pfr.parse_stats(tbody)
    assert result["team_total_yds"] == "400"
    assert result["team_points"] == "27"
    assert result["opp_total_yds"] == "350"
    assert result["opp_points"] == "24"

def test_parse_stats_missing_data_stat():
    html = '''
    <tbody>
        <tr><th>Team Stats</th><td>Not a stat</td></tr>
        <tr><th>Opp. Stats</th><td data-stat="total_yds">350</td></tr>
    </tbody>
    '''
    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody")
    result = pfr.parse_stats(tbody)
    assert "opp_total_yds" in result
    assert "team_" not in result  # stat without data-stat should be skipped

def test_parse_stats_empty_value():
    html = '''
    <tbody>
        <tr><th>Team Stats</th><td data-stat="total_yds"></td></tr>
        <tr><th>Opp. Stats</th><td data-stat="total_yds">350</td></tr>
    </tbody>
    '''
    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody")
    result = pfr.parse_stats(tbody)
    assert "team_total_yds" not in result
    assert result["opp_total_yds"] == "350"

def test_parse_stats_malformed_html():
    html = '''
    <tbody>
        <tr><td data-stat="total_yds">400</td></tr>
        <tr><th>Opp. Stats</th><td data-stat="total_yds">350</td></tr>
    </tbody>
    '''
    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody")
    result = pfr.parse_stats(tbody)
    assert result == {"opp_total_yds": "350"}  # only opp. stats row should be parsed because the first row has no <th>


def test_parse_team_totals():
    html = '''<tfoot><tr><td data-stat="xpa">10</td><td data-stat="xpm">8</td></tr></tfoot>'''
    wrapper_html = f"<table>{html}</table>"
    soup = BeautifulSoup(wrapper_html, "html.parser")
    tfoot = soup.find("tfoot")
    result = pfr.parse_team_totals(tfoot)
    assert result == {"team_total_xpa": "10", "team_total_xpm": "8"}


def test_construct_player_urls_with_hash():
    players = [{"player_name": "Tom Brady", "position": "QB", "hashed_name": "B/BradTo00", "pfr_available": 1}]
    result = pfr.construct_player_urls(players, 2023)
    assert result[0]["url"].endswith("/gamelog/2023")
    assert "Tom Brady" in result[0]["player"]


def test_parse_player_and_team_totals():
    html = '''
    <table>
        <tbody>
            <tr>
                <td data-stat="name_display"><a>Tom Brady</a></td>
                <td data-stat="pass_td">3</td>
            </tr>
        </tbody>
    </table>
    <tfoot><tr><td data-stat="pass_cmp">20</td></tr></tfoot>
    '''
    soup = BeautifulSoup(html, "html.parser")
    players_table = soup.find("table")
    team_totals = soup.find("tfoot")
    with patch("scrape_and_score.scraping.pfr.pfr_utils.player_service.normalize_name", return_value="tom-brady"):
        player_metrics, team_metrics = pfr.parse_player_and_team_totals(players_table, team_totals)
    assert player_metrics["tom-brady"]["pass_td"] == "3"
    assert team_metrics["team_total_pass_cmp"] == "20"
