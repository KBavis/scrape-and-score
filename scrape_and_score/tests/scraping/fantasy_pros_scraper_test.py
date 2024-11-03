from unittest.mock import patch
from bs4 import BeautifulSoup
from scraping import fantasy_pros_scraper 


soup = None

def setUpModule():
    """Module-level setup for initializing soup with sample HTML."""
    global soup
    html_content = """
    <div class="team-list">
        <input class="team-name" value="New England Patriots" />
        <div class="position-list">
            <h4 class="position-head">ECR Quarterbacks</h4>
            <a class="fp-player-link" fp-player-name="Mac Jones"></a>
        </div>
        <div class="position-list">
            <h4 class="position-head">ECR Running Backs</h4>
            <a class="fp-player-link" fp-player-name="Rhamondre Stevenson"></a>
            <a class="fp-player-link" fp-player-name="Ezekiel Elliott"></a>
        </div>
    </div>
    <div class="team-list">
        <input class="team-name" value="Kansas City Chiefs" />
        <div class="position-list">
            <h4 class="position-head">ECR Quarterbacks</h4>
            <a class="fp-player-link" fp-player-name="Patrick Mahomes"></a>
        </div>
        <div class="position-list">
            <h4 class="position-head">ECR Tight Ends</h4>
            <a class="fp-player-link" fp-player-name="Travis Kelce"></a>
        </div>
    </div>
    """
    # Parse HTML content with BeautifulSoup and assign to the global variable
    soup = BeautifulSoup(html_content, "html.parser")


@patch('scraping.util.fetch_page')
@patch('scraping.fantasy_pros_scraper.fetch_team_data')
def test_scrape(mock_fetch_team_data, mock_fetch_page):
   # arrange 
   url = "https://fantasypros.com"
   mock_html = "<html><body></body></html>"
   mock_fetch_page.return_value = mock_html
   expected_values = [
         {'team': 'Team A', 'position': 'QB', 'player_name': 'Player 1'},
         {'team': 'Team A', 'position': 'QB', 'player_name': 'Player 2'}
   ]
   mock_fetch_team_data.return_value = expected_values
   
   # act 
   result = fantasy_pros_scraper.scrape(url)
   
   # assert 
   assert result == expected_values
   
   
   

def test_fetch_team_data():
    
    # arrange 
    expected_output = [
        {'team': 'New England Patriots', 'position': 'QB', 'player_name': 'Mac Jones'},
        {'team': 'New England Patriots', 'position': 'RB', 'player_name': 'Rhamondre Stevenson'},
        {'team': 'New England Patriots', 'position': 'RB', 'player_name': 'Ezekiel Elliott'},
        {'team': 'Kansas City Chiefs', 'position': 'QB', 'player_name': 'Patrick Mahomes'},
        {'team': 'Kansas City Chiefs', 'position': 'TE', 'player_name': 'Travis Kelce'},
    ]

    # act 
    actual_output = fantasy_pros_scraper.fetch_team_data(soup)
    
    # assert 
    assert actual_output == expected_output, f"Expected {expected_output}, got {actual_output}"
    