from unittest.mock import patch
from bs4 import BeautifulSoup
from scraping import fantasy_pros_scraper 


@patch('scraping.fantasy_pros_scraper.construct_url', return_value = 'URL')
@patch('scraping.fantasy_pros_scraper.fetch_page', return_value = None)
@patch('scraping.fantasy_pros_scraper.get_depth_chart')
@patch('scraping.fantasy_pros_scraper.BeautifulSoup')
def test_scrape_returns_expected_data(mock_soup, mock_get_depth_chart, mock_fetch_page, mock_construct_url): 
    base_url = 'https://www.fantasypros.com/nfl/depth-chart/{TEAM}.php'
    teams = [{'name': 'Indianapolis Colts', 'team_id': 20}]
    expected_data = [{'player_name': 'Anthony Richardson', 'position': 'QB', 'team': 'Indianapolis Colts'}]
    mock_get_depth_chart.return_value = expected_data
    mock_soup.return_value = None
    
    data = fantasy_pros_scraper.scrape(base_url, teams)
    
    assert data == expected_data

@patch('scraping.fantasy_pros_scraper.construct_url', return_value = 'URL')
@patch('scraping.fantasy_pros_scraper.fetch_page', return_value = None)
@patch('scraping.fantasy_pros_scraper.get_depth_chart')
@patch('scraping.fantasy_pros_scraper.BeautifulSoup')
def test_scrape_calls_expected_functions(mock_soup, mock_get_depth_chart, mock_fetch_page, mock_construct_url): 
    base_url = 'https://www.fantasypros.com/nfl/depth-chart/{TEAM}.php'
    teams = [{'name': 'Indianapolis Colts', 'team_id': 20}]
    expected_data = [{'player_name': 'Anthony Richardson', 'position': 'QB', 'team': 'Indianapolis Colts'}]
    mock_get_depth_chart.return_value = expected_data
    mock_soup.return_value = None
    
    fantasy_pros_scraper.scrape(base_url, teams)
    
    mock_soup.assert_called_once()
    mock_get_depth_chart.assert_called_once() 
    mock_construct_url.assert_called_once() 
    mock_fetch_page.assert_called_once()
    
    
    
def test_construct_url_constructs_correct_url(): 
    base_url = 'https://www.fantasypros.com/nfl/depth-chart/{TEAM}.php'
    expected_url = 'https://www.fantasypros.com/nfl/depth-chart/arizona-cardinals.php'
    team = 'Arizona Cardinals'
    
    actual_url = fantasy_pros_scraper.construct_url(team, base_url)
    
    assert actual_url == expected_url

    
def test_get_depth_chart_returns_expected_depth_chart():
    html = """
    <html>
        <body>
            <table>
                <tr>
                    <td>QB</td>
                    <td>Anthony Richardson</td>
                </tr>
                <tr>
                    <td>RB</td>
                    <td>Jonathan Taylor</td>
                </tr>
            </table>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, 'html.parser')
    team = "Indianapolis Colts"
    team_id = 20
    
    expected_data = [
        {"player_name": "Anthony Richardson", "position": "QB", "team_id": team_id},
        {"player_name": "Jonathan Taylor", "position": "RB", "team_id": team_id}
    ]
    
    result = fantasy_pros_scraper.get_depth_chart(soup, team, team_id)
    assert result == expected_data
    
    

def test_get_depth_chart_skips_rows_without_name_and_position():
    html = """
    <html>
        <body>
            <table>
                <tr>
                    <td>QB</td>
                    <td>Anthony Richardson</td>
                </tr>
                <tr>
                    <td></td>
                    <td></td>
                </tr>
                <tr>
                    <td>RB</td>
                    <td>Jonathan Taylor</td>
                </tr>
                <tr>
                    <td>QB</td>
                </tr>
            </table>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, 'html.parser')
    team = "Indianapolis Colts"
    team_id = 20
    
    # expected data doesn't include empty cells
    expected_data = [
        {"player_name": "Anthony Richardson", "position": "QB", "team_id": team_id},
        {"player_name": "Jonathan Taylor", "position": "RB", "team_id": team_id}
    ]
    
    result = fantasy_pros_scraper.get_depth_chart(soup, team, team_id)
    assert result == expected_data
