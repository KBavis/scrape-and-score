from unittest.mock import patch, MagicMock
from scraping import pfr_scraper
import pandas as pd
from datetime import date, datetime, timedelta
import pytest
from scraping_helper import mock_find_common_metrics, mock_find_wr_metrics, \
         mock_find_rb_metrics, mock_find_qb_metrics, setup_game_log_mocks, \
         mock_add_common_game_log_metrics, mock_add_wr_game_log_metrics, \
         setup_get_href_mocks, mock_get_href_response, mocked_extract_int  

def test_extract_float_returns_zero_when_none():
   tr_mock = MagicMock()
   
   tr_mock.find.return_value = None
   
   result = pfr_scraper.extract_float(tr_mock, "rush-att")
   
   assert result == 0.0
  

def test_extract_float_returns_zero_when_empty():
   tr_mock = MagicMock()
   text_mock = MagicMock()
   text_mock.text = ''
   
   tr_mock.find.return_value = text_mock
   
   result = pfr_scraper.extract_float(tr_mock, "rush-att")
   
   assert result == 0.0
   

def test_extract_float_returns_float():
   tr_mock = MagicMock()
   text_mock = MagicMock()
   text_mock.text = "12.40"
   
   
   tr_mock.find.return_value = text_mock
   
   result = pfr_scraper.extract_float(tr_mock, "rush-att")
   
   assert isinstance(result, float)

def test_extract_float_returns_correct_value():
   tr_mock = MagicMock()
   
   text_mock = MagicMock()
   text_mock.text = "12.40"
   
   
   tr_mock.find.return_value = text_mock
   
   result = pfr_scraper.extract_float(tr_mock, "rush-att")
   
   assert result == 12.40 
   

def test_extract_int_returns_zero_when_none():
   tr_mock = MagicMock()
   
   tr_mock.find.return_value = None
   
   result = pfr_scraper.extract_int(tr_mock, "rush-att")
   
   assert result == 0.0
  

def test_extract_int_returns_zero_when_empty():
   tr_mock = MagicMock()
   text_mock = MagicMock()
   text_mock.text = ''
   
   tr_mock.find.return_value = text_mock
   
   result = pfr_scraper.extract_int(tr_mock, "rush-att")
   
   assert result == 0.0
   

def test_extract_int_returns_int():
   tr_mock = MagicMock()
   text_mock = MagicMock()
   text_mock.text = "12"
   
   
   tr_mock.find.return_value = text_mock
   
   result = pfr_scraper.extract_int(tr_mock, "rush-att")
   
   assert isinstance(result, int)


def test_extract_int_returns_correct_value():
   tr_mock = MagicMock()
   text_mock = MagicMock()
   text_mock.text = "12"
   
   
   tr_mock.find.return_value = text_mock
   
   result = pfr_scraper.extract_int(tr_mock, "rush-att")
   
   assert result == 12 
   

def test_get_additional_metrics_for_qb():
   expected_additional_fields = {
      'cmp': [],
      'att': [],
      'pass_yds': [],
      'pass_td': [],
      'int': [],
      'rating': [],
      'sacked': [],
      'rush_att': [],
      'rush_yds': [],
      'rush_td': [],
   }
   
   result = pfr_scraper.get_additional_metrics('QB')
   
   assert result == expected_additional_fields
   
   
def test_get_additional_metrics_for_rb():
   expected_additional_fields = {
      'rush_att': [],
      'rush_yds': [],
      'rush_td': [],
      'tgt': [],
      'rec': [],
      'rec_yds': [],
      'rec_td': [],
   }
   
   result = pfr_scraper.get_additional_metrics('RB')
   
   assert result == expected_additional_fields
   
def test_get_additional_metrics_for_wr():
   expected_additional_fields = {
      'tgt': [],
      'rec': [],
      'rec_yds': [],
      'rec_td': [],
      'snap_pct': [],
   } 
   
   result = pfr_scraper.get_additional_metrics('WR')
   
   assert result == expected_additional_fields
  
def test_get_additional_metrics_for_te():
   expected_additional_fields = {
      'tgt': [],
      'rec': [],
      'rec_yds': [],
      'rec_td': [],
      'snap_pct': [],
   }
   
   result = pfr_scraper.get_additional_metrics('TE')
   
   assert result == expected_additional_fields


def test_get_additional_metrics_for_invalid_position():
   
   with pytest.raises(Exception, match="The position 'OL' is not a valid position to fetch metrics for."):
      pfr_scraper.get_additional_metrics('OL')


def test_add_common_game_log_metrics():
   # arrange dummy dictionary
   data = {
      'date': [],
      'week': [],
      'team': [],
      'game_location': [],
      'opp': [],
      'result': [],
      'team_pts': [],
      'opp_pts': [],
   }
   
   # set up mocks 
   tr_mock = MagicMock()
   tr_mock.find.side_effect = mock_find_common_metrics #update tr mock to utilize our mock_find instead of find() method
   
   pfr_scraper.add_common_game_log_metrics(data, tr_mock)
   
   assert data['date'] == ['2023-09-12']
   assert data['week'] == [1]
   assert data['team'] == ['DAL']
   assert data['game_location'] == ['@']
   assert data['opp'] == ['NYG']
   assert data['result'] == ['W']
   assert data['team_pts'] == [20]
   assert data['opp_pts'] == [10]
   

def test_add_wr_specific_game_log_metrics():
   # arrange dummy dictionary
   data = {
      'tgt': [],
      'rec': [],
      'rec_yds': [],
      'rec_td': [],
      'snap_pct': [],
   }
   
   # set up mocks 
   tr_mock = MagicMock() 
   tr_mock.find.side_effect = mock_find_wr_metrics
   
   # invoke 
   pfr_scraper.add_wr_specific_game_log_metrics(data, tr_mock)
   
   # assert
   assert data['tgt'] == [9]
   assert data['rec'] == [6]
   assert data['rec_yds'] == [118]
   assert data['rec_td'] == [2]
   assert data['snap_pct'] == [.67]

def test_add_rb_specific_game_log_metrics():
   # arrange dummy dictionary
   data = {
      'rush_att': [],
      'rush_yds': [],
      'rush_td': [],
      'tgt': [],
      'rec': [],
      'rec_yds': [],
      'rec_td': []
   }   
   
   # setup mocks
   tr_mock = MagicMock()
   tr_mock.find.side_effect = mock_find_rb_metrics
   
   # act 
   pfr_scraper.add_rb_specific_game_log_metrics(data, tr_mock)
   
   # assert 
   assert data['rush_att'] == [9]
   assert data['rush_yds'] == [68]
   assert data['rush_td'] == [2]
   assert data['tgt'] == [2]
   assert data['rec'] == [2]
   assert data['rec_yds'] == [41]
   assert data['rec_td'] == [0]
   
   
def test_add_qb_specific_game_log_metrics():
   # arrange dummy dictionary
   data = {
      'cmp': [],
      'att': [],
      'pass_yds': [],
      'pass_td': [],
      'int': [],
      'rating': [],
      'sacked': [],
      'rush_att': [],
      'rush_yds': [],
      'rush_td': []
   }

   # setup mocks
   tr_mock = MagicMock()
   tr_mock.find.side_effect = mock_find_qb_metrics

   # act
   pfr_scraper.add_qb_specific_game_log_metrics(data, tr_mock)

   # assert
   assert data['cmp'] == [24]
   assert data['att'] == [36]
   assert data['pass_yds'] == [315]
   assert data['pass_td'] == [3]
   assert data['int'] == [1]
   assert data['rating'] == [98.7]
   assert data['sacked'] == [2]
   assert data['rush_att'] == [5]
   assert data['rush_yds'] == [23]
   assert data['rush_td'] == [1]
   

@patch('scraping.pfr_scraper.get_additional_metrics')
@patch('scraping.pfr_scraper.add_common_game_log_metrics')
@patch('scraping.pfr_scraper.add_qb_specific_game_log_metrics')
def test_get_game_log_for_qb_calls_expected_functions(mock_add_qb_metrics, mock_add_common_metrics, mock_get_additional_metrics):
   mock_soup = setup_game_log_mocks('Valid')

   pfr_scraper.get_game_log(mock_soup, 'QB')
   
   mock_add_common_metrics.assert_called_once()
   mock_add_qb_metrics.assert_called_once()
   mock_get_additional_metrics.assert_called_once()
   

@patch('scraping.pfr_scraper.get_additional_metrics')
@patch('scraping.pfr_scraper.add_common_game_log_metrics')
@patch('scraping.pfr_scraper.add_wr_specific_game_log_metrics')
def test_get_game_log_for_wr_calls_expected_functions(mock_add_wr_metrics, mock_add_common_metrics, mock_get_additional_metrics):
   mock_soup = setup_game_log_mocks('Valid')
   
   pfr_scraper.get_game_log(mock_soup, 'WR')
   
   mock_add_common_metrics.assert_called_once()
   mock_add_wr_metrics.assert_called_once()
   mock_get_additional_metrics.assert_called_once()
   

@patch('scraping.pfr_scraper.get_additional_metrics')
@patch('scraping.pfr_scraper.add_common_game_log_metrics')
@patch('scraping.pfr_scraper.add_rb_specific_game_log_metrics')
def test_get_game_log_for_rb_calls_expected_functions(mock_add_rb_metrics, mock_add_common_metrics, mock_get_additional_metrics):
   mock_soup = setup_game_log_mocks('Valid')
   
   pfr_scraper.get_game_log(mock_soup, 'RB')
   
   mock_add_common_metrics.assert_called_once()
   mock_add_rb_metrics.assert_called_once()
   mock_get_additional_metrics.assert_called_once()


@patch('scraping.pfr_scraper.get_additional_metrics')
@patch('scraping.pfr_scraper.add_common_game_log_metrics')
@patch('scraping.pfr_scraper.add_wr_specific_game_log_metrics')
def test_get_game_log_for_te_calls_expected_functions(mock_add_wr_metrics, mock_add_common_metrics, mock_get_additional_metrics): 
    
   mock_soup = setup_game_log_mocks('Valid')
   
   pfr_scraper.get_game_log(mock_soup, 'TE')
   
   mock_add_common_metrics.assert_called_once()
   mock_add_wr_metrics.assert_called_once()
   mock_get_additional_metrics.assert_called_once()


@patch('scraping.pfr_scraper.get_additional_metrics')
@patch('scraping.pfr_scraper.add_common_game_log_metrics')
@patch('scraping.pfr_scraper.add_wr_specific_game_log_metrics')
def test_get_game_log_ignores_inactive_status(mock_add_wr_metrics, mock_add_common_metrics, mock_get_additional_metrics): 
   mock_soup = setup_game_log_mocks('Inactive') # setup mocks with inactive status
   mock_get_additional_metrics.return_value = {'tgt': [],'rec': [],'rec_yds': [],'rec_td': [],'snap_pct': []}
  
   pandas_df = pfr_scraper.get_game_log(mock_soup, 'WR')
   
   assert pandas_df.empty
   
   
@patch('scraping.pfr_scraper.get_additional_metrics')
@patch('scraping.pfr_scraper.add_common_game_log_metrics')
@patch('scraping.pfr_scraper.add_wr_specific_game_log_metrics')
def test_get_game_log_ignores_inactive_status(mock_add_wr_metrics, mock_add_common_metrics, mock_get_additional_metrics): 
   mock_soup = setup_game_log_mocks('Did Not Play') # setup mocks with inactive status
   mock_get_additional_metrics.return_value = {'tgt': [],'rec': [],'rec_yds': [],'rec_td': [],'snap_pct': []}
  
   pandas_df = pfr_scraper.get_game_log(mock_soup, 'WR')
   
   assert pandas_df.empty
   
   
@patch('scraping.pfr_scraper.get_additional_metrics')
@patch('scraping.pfr_scraper.add_common_game_log_metrics')
@patch('scraping.pfr_scraper.add_wr_specific_game_log_metrics')
def test_get_game_log_ignores_inactive_status(mock_add_wr_metrics, mock_add_common_metrics, mock_get_additional_metrics): 
   mock_soup = setup_game_log_mocks('Injured Reserve') # setup mocks with inactive status
   mock_get_additional_metrics.return_value = {'tgt': [],'rec': [],'rec_yds': [],'rec_td': [],'snap_pct': []}
  
   pandas_df = pfr_scraper.get_game_log(mock_soup, 'WR')
   
   assert pandas_df.empty
   

@patch('scraping.pfr_scraper.get_additional_metrics')
@patch('scraping.pfr_scraper.add_common_game_log_metrics')
@patch('scraping.pfr_scraper.add_wr_specific_game_log_metrics')
def test_get_game_log_returns_expected_df(mock_add_wr_metrics, mock_add_common_metrics, mock_get_additional_metrics):
   mock_add_wr_metrics.side_effect = mock_add_wr_game_log_metrics
   mock_add_common_metrics.side_effect = mock_add_common_game_log_metrics
   
   mock_soup = setup_game_log_mocks('Valid')
   mock_get_additional_metrics.return_value = {'tgt': [],'rec': [],'rec_yds': [],'rec_td': [],'snap_pct': []}
   
   pandas_df = pfr_scraper.get_game_log(mock_soup, 'WR')
   
   expected_data = {
      'date': ['2024-11-10'],
      'week': [10],
      'team': ['Team A'],
      'game_location': ['@'],
      'opp': ['Team B'],
      'result': ['W'],
      'team_pts': [24],
      'opp_pts': [17],
      'tgt': [7],
      'rec': [5],
      'rec_yds': [102],
      'rec_td': [1],
      'snap_pct': [67.7]
   }
   expected_df = pd.DataFrame(data=expected_data)
   
   pd.testing.assert_frame_equal(pandas_df, expected_df)
   
   

@patch('scraping.pfr_scraper.fuzz.partial_ratio')
def test_check_name_similarity_parses_correct_name(mock_partial_ratio):
   player_text = "Anthony Richardson Jr."
   player_name = "Anthony Richardson"
   
   pfr_scraper.check_name_similarity(player_text, player_name)
   
   mock_partial_ratio.assert_called_once_with("Anthony Richardson", player_name)
   

@patch('scraping.pfr_scraper.check_name_similarity')
def test_get_href_skips_players_with_invalid_years(mock_check_name_similarity):
   player_name = 'Anthony Richardson'
   year = 2024 
   position = 'QB'
   
   # setup mocks  
   mock_soup = setup_get_href_mocks(True, False)
   
   href = pfr_scraper.get_href(player_name, position, year, mock_soup)
   
   mock_check_name_similarity.assert_not_called() 
   assert href == None


@patch('scraping.pfr_scraper.check_name_similarity')
def test_get_href_skips_players_not_active_within_specified_year(mock_check_name_similarity):
   player_name = 'Anthony Richardson'
   year = 2021 # use year with no valid players
   position = 'QB'
   
   # setup mocks 
   mock_check_name_similarity.return_value = 95 # valid name similarity
   mock_soup = setup_get_href_mocks(False, False)
   
   href = pfr_scraper.get_href(player_name, position, year, mock_soup)
   
   assert href == None
   
   
@patch('scraping.pfr_scraper.check_name_similarity')
def test_get_href_skips_players_not_active_at_specified_position(mock_check_name_similarity):
   player_name = 'Anthony Richardson'
   year = 2023
   position = 'RB' # invalid position 
   
   # setup mocks 
   mock_check_name_similarity.return_value = 95 # valid name similarity
   mock_soup = setup_get_href_mocks(False, False)
   
   href = pfr_scraper.get_href(player_name, position, year, mock_soup)
   
   assert href == None


@patch('scraping.pfr_scraper.check_name_similarity')
def test_get_href_skips_players_if_name_isnt_similar_enough(mock_check_name_similarity):
   player_name = 'Anthony Richardson'
   year = 2023
   position = 'QB' 
   
   # setup mocks 
   mock_check_name_similarity.return_value = 85 # invalid similarity
   mock_soup = setup_get_href_mocks(False, False)
   
   href = pfr_scraper.get_href(player_name, position, year, mock_soup)
   
   assert href == None


@patch('scraping.pfr_scraper.check_name_similarity')
def test_get_href_returns_none_when_no_a_tag(mock_check_name_similarity):
   player_name = 'Anthony Richardson'
   year = 2023
   position = 'QB' 
   
   # setup mocks 
   mock_check_name_similarity.return_value = 95 # valid similarity
   mock_soup = setup_get_href_mocks(False, True)
   
   href = pfr_scraper.get_href(player_name, position, year, mock_soup)
   
   assert href == None


@patch('scraping.pfr_scraper.check_name_similarity')
def test_get_href_returns_expected_href(mock_check_name_similarity):
   player_name = 'Anthony Richardson'
   year = 2023
   position = 'QB' 
   
   # setup mocks 
   mock_check_name_similarity.return_value = 95 # valid similarity
   mock_soup = setup_get_href_mocks(False, False)
   
   href = pfr_scraper.get_href(player_name, position, year, mock_soup)
   
   assert href == 'my-href'
   

@patch('scraping.pfr_scraper.get_href')
@patch('scraping.pfr_scraper.fetch_page')
def test_get_player_urls_returns_expected_urls(mock_fetch_page, mock_get_href):
   # arrange 
   ordered_players = { "A": [], "R" : [ {"team": "Indianapolis Colts", "position": "QB", "player_name": "Anthony Richardson"}], 
                        "Z": [{"team": "Indianapolis Colts", "position": "RB", "player_name": "Test Ziegler"}, {"team": "Tennessee Titans", "position": "RB", "player_name": "Xavier Zegette"}]}
   year = 2024
   expected_urls = [
      {"player": "Anthony Richardson", "position": "QB", "url": "https://www.pro-football-reference.com/ARich/gamelog/2024"},
      {"player": "Test Ziegler", "position": "RB", "url": "https://www.pro-football-reference.com/TZieg/gamelog/2024"},
      {"player": "Xavier Zegette", "position": "RB", "url": "https://www.pro-football-reference.com/XZeg/gamelog/2024"},
   ]
   
   
   # setup mocks 
   mock_fetch_page.return_value = "<html><body><h1>Testing</h1></body></html>"
   mock_get_href.side_effect = mock_get_href_response
   
   # act 
   actual_urls = pfr_scraper.get_player_urls(ordered_players, year)
   
   # assert 
   assert expected_urls == actual_urls
   


@patch('scraping.pfr_scraper.get_href')
@patch('scraping.pfr_scraper.fetch_page')
def test_get_player_urls_calls_expected_functions(mock_fetch_page, mock_get_href): 
   # arrange 
   ordered_players = { "A": [], "R" : [ {"team": "Indianapolis Colts", "position": "QB", "player_name": "Anthony Richardson"}], 
                        "Z": [{"team": "Indianapolis Colts", "position": "RB", "player_name": "Test Ziegler"}, {"team": "Tennessee Titans", "position": "RB", "player_name": "Xavier Zegette"}]}
   year = 2024
   
   
   # setup mocks 
   mock_fetch_page.return_value = "<html><body><h1>Testing</h1></body></html>"
   mock_get_href.side_effect = mock_get_href_response
   
   # act 
   pfr_scraper.get_player_urls(ordered_players, year)
   
   # assert 
   assert mock_fetch_page.call_count == 3
   assert mock_get_href.call_count == 3
   
   

@patch('scraping.pfr_scraper.get_href')
@patch('scraping.pfr_scraper.fetch_page')
def test_get_player_urls_skips_appending_urls_if_none_found(mock_fetch_page, mock_get_href): 
   # arrange 
   ordered_players = { "A": [], "R" : [ {"team": "Indianapolis Colts", "position": "QB", "player_name": "Anthony Richardson"}], 
                        "Z": [{"team": "Indianapolis Colts", "position": "RB", "player_name": "Test Ziegler"}, {"team": "Tennessee Titans", "position": "RB", "player_name": "Xavier Zegette"}]}
   year = 2024
   
   
   # setup mocks 
   mock_fetch_page.return_value = "<html><body><h1>Testing</h1></body></html>"
   mock_get_href.return_value = None # have get_href fail to find href 
   
   # act 
   actual_urls = pfr_scraper.get_player_urls(ordered_players, year)
   
   # assert 
   assert actual_urls == []
   
   

def test_order_players_by_last_name_successfully_orders_players():
   # arrange
   player_data = [{"player_name": "Chris Cranger"}, {"player_name":  "Order Amega"}, {"player_name": "Zebra Zilch"}, {"player_name": "Brandon Baker"}, {"player_name" : "Harvey Yonkers"}]
   expected_ordered_players = {
        "A": [],"B": [],"C": [],"D": [],"E": [],
        "F": [],"G": [],"H": [],"I": [],"J": [],
        "K": [],"L": [],"M": [],"N": [],"O": [],
        "P": [],"Q": [],"R": [],"S": [],"T": [],
        "U": [],"V": [],"W": [],"X": [],"Y": [],
        "Z": []
    }
   expected_ordered_players['C'].append({"player_name": "Chris Cranger"})
   expected_ordered_players['A'].append({"player_name": "Order Amega"})
   expected_ordered_players['Y'].append({"player_name": "Harvey Yonkers"})
   expected_ordered_players['B'].append({"player_name": "Brandon Baker"})
   expected_ordered_players['Z'].append({"player_name": "Zebra Zilch"})
   
   # act
   actual_ordered_players = pfr_scraper.order_players_by_last_name(player_data)
   
   # assert 
   assert expected_ordered_players == actual_ordered_players


def test_get_game_date_prior_to_new_year_when_game_date_prior_to_new_year():
   #arrange 
   current_date = date(2023, 8, 20) # set date to be during August, prior to New Year
   expected_game_date = date(2023, 11, 20) # game date year should be for current year
   
   # setup mocks 
   game = MagicMock() 
   game_found = MagicMock() 
   game_found.text = 'November 20' # set game month to be Novemeber 
   game.find.return_value = game_found
   
   
   # act 
   actual_game_date = pfr_scraper.get_game_date(game, current_date)
   
   # assert 
   assert actual_game_date == expected_game_date
   
   

def test_get_game_date_after_new_year_when_game_date_prior_to_new_year():
   #arrange 
   current_date = date(2023, 1, 20) # set date to be during January, after New Year
   expected_game_date = date(2023, 11, 20) # game date year should be for current year
   
   # setup mocks 
   game = MagicMock() 
   game_found = MagicMock() 
   game_found.text = 'November 20'
   game.find.return_value = game_found
   
   
   # act 
   actual_game_date = pfr_scraper.get_game_date(game, current_date)
   
   # assert 
   assert actual_game_date == expected_game_date
   
 
def test_get_game_date_after_new_year_when_game_date_after_new_year():
   #arrange 
   current_date = date(2024, 1, 10) # set date to be during January, after New Year
   expected_game_date = date(2024, 1, 20) # set game date to be within January/Feb
   
   # setup mocks 
   game = MagicMock() 
   game_found = MagicMock() 
   game_found.text = 'January 20'
   game.find.return_value = game_found
   
   
   # act 
   actual_game_date = pfr_scraper.get_game_date(game, current_date)
   
   # assert 
   assert actual_game_date == expected_game_date
   

def test_get_game_date_before_new_year_when_game_date_after_new_year():
   #arrange 
   current_date = date(2023, 9, 20) # set date to be during Septmeber, prior to New Year
   expected_game_date = date(2024, 1, 20) # set game date to be within January/Feb
   
   # setup mocks 
   game = MagicMock() 
   game_found = MagicMock() 
   game_found.text = 'January 20'
   game.find.return_value = game_found
   
   
   # act 
   actual_game_date = pfr_scraper.get_game_date(game, current_date)
   
   # assert 
   assert actual_game_date == expected_game_date


def test_calculate_distance_returns_expected_distance():
   # arrange 
   city1 = {'latitude': 42.3656, 'longitude': 71.0096, 'airport': 'BOS'}
   city2 = {'latitude': 33.4352, 'longitude': 112.0101, 'airport': 'PHX'}
   expected_distance = 2294.86
   
   # act
   actual_distance = pfr_scraper.calculate_distance(city1, city2)
   
   assert expected_distance == round(actual_distance, ndigits=2)


@patch('scraping.pfr_scraper.fetch_page')
def test_get_team_metrics_html_calls_expected_function(mock_fetch_page):
   # arrange 
   team_name = 'Carolina Panthers'
   year = 2024
   template_url = "https://www.pro-football-reference.com/teams/{TEAM_ACRONYM}/{CURRENT_YEAR}.htm"
   expected_html = "<html><body><h1>h1</h1></body></body>"
   
   # mock 
   mock_fetch_page.return_value = expected_html
   
   # act
   pfr_scraper.get_team_metrics_html(team_name, year, template_url)
   
   # assert 
   mock_fetch_page.assert_called_with("https://www.pro-football-reference.com/teams/car/2024.htm")
   

@patch('scraping.pfr_scraper.fetch_page')
def test_get_team_metrics_html_returns_expected_response(mock_fetch_page):
   # arrange 
   team_name = 'Carolina Panthers'
   year = 2024
   template_url = "https://www.pro-football-reference.com/teams/{TEAM_ACRONYM}/{CURRENT_YEAR}.htm"
   expected_html = "<html><body><h1>h1</h1></body></body>"
   
   # mock 
   mock_fetch_page.return_value = expected_html
   
   # act
   actual_html = pfr_scraper.get_team_metrics_html(team_name, year, template_url)
   
   # assert 
   assert actual_html == expected_html


@patch('scraping.pfr_scraper.extract_int')
def test_calculate_yardage_totals_returns_expected_tot_yds(mock_extract_int):
   games = [MagicMock()]
   index = 0
   mock_extract_int.side_effect = mocked_extract_int
   
   # act 
   tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds = pfr_scraper.calculate_yardage_totals(games, index)
   
   # assert 
   assert tot_yds == 10

   
@patch('scraping.pfr_scraper.extract_int')
def test_calculate_yardage_totals_returns_expected_pass_yds(mock_extract_int):
   games = [MagicMock()]
   index = 0
   mock_extract_int.side_effect = mocked_extract_int
   
   # act 
   tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds = pfr_scraper.calculate_yardage_totals(games, index)
   
   # assert 
   assert pass_yds == 20
  
   
@patch('scraping.pfr_scraper.extract_int')
def test_calculate_yardage_totals_returns_expected_rush_yds(mock_extract_int):
   games = [MagicMock()]
   index = 0
   mock_extract_int.side_effect = mocked_extract_int
   
   # act 
   tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds = pfr_scraper.calculate_yardage_totals(games, index)
   
   # assert 
   assert rush_yds == 30
   

@patch('scraping.pfr_scraper.extract_int')
def test_calculate_yardage_totals_returns_expected_opp_tot_yds(mock_extract_int):
   games = [MagicMock()]
   index = 0
   mock_extract_int.side_effect = mocked_extract_int
   
   # act 
   tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds = pfr_scraper.calculate_yardage_totals(games, index)
   
   # assert 
   assert opp_tot_yds == 40


@patch('scraping.pfr_scraper.extract_int')
def test_calculate_yardage_totals_returns_expected_opp_pass_yds(mock_extract_int):
   games = [MagicMock()]
   index = 0
   mock_extract_int.side_effect = mocked_extract_int
   
   # act 
   tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds = pfr_scraper.calculate_yardage_totals(games, index)
   
   # assert 
   assert opp_pass_yds == 50


@patch('scraping.pfr_scraper.extract_int')
def test_calculate_yardage_totals_returns_expected_opp_rush_yds(mock_extract_int):
   games = [MagicMock()]
   index = 0
   mock_extract_int.side_effect = mocked_extract_int
   
   # act 
   tot_yds, pass_yds, rush_yds, opp_tot_yds, opp_pass_yds, opp_rush_yds = pfr_scraper.calculate_yardage_totals(games, index)
   
   # assert 
   assert opp_rush_yds == 60


@patch('scraping.pfr_scraper.extract_int')
def test_calculate_yardage_totals_calls_expected_functions(mock_extract_int):
   games = [MagicMock()]
   index = 0
   mock_extract_int.side_effect = mocked_extract_int
   
   # act 
   pfr_scraper.calculate_yardage_totals(games, index)
   
   # assert 
   assert mock_extract_int.call_count == 6


def test_calculate_rest_days_when_index_is_zero():
   games = MagicMock()
   index = 0 # set index to zero
   year = 2024 
   expected_rest_days = 10
   
   actual_rest_days = pfr_scraper.calculate_rest_days(games, index, year)
   
   assert expected_rest_days == actual_rest_days


def test_calculate_rest_days_when_both_games_prior_to_new_year():
   previous_game_date = MagicMock()
   previous_game_date.text = 'September 21' # set date prior to new year
   previous_game = MagicMock()
   previous_game.find.return_value = previous_game_date
     
   current_game_date = MagicMock()
   current_game_date.text = 'September 28' # set date prior to new year
   current_game = MagicMock() 
   current_game.find.return_value = current_game_date
   
   games = [previous_game, current_game]
   
   index = 1
   year = 2024 
   expected_rest_days = date(2024, 9, 28) - date(2024, 9, 21)
   
   actual_rest_days = pfr_scraper.calculate_rest_days(games, index, year)
   
   assert expected_rest_days == actual_rest_days


def test_calculate_rest_days_when_both_games_in_new_year():
   previous_game_date = MagicMock()
   previous_game_date.text = 'January 7'
   previous_game = MagicMock()
   previous_game.find.return_value = previous_game_date
     
   current_game_date = MagicMock()
   current_game_date.text = 'January 14'
   current_game = MagicMock() 
   current_game.find.return_value = current_game_date
   
   games = [previous_game, current_game]
   
   index = 1
   year = 2024 
   expected_rest_days = date(2025, 1, 14) - date(2025, 1, 7)
   
   actual_rest_days = pfr_scraper.calculate_rest_days(games, index, year)
   
   assert expected_rest_days == actual_rest_days


def test_calculate_rest_days_when_current_game_only_in_new_year():
   previous_game_date = MagicMock()
   previous_game_date.text = 'December 31'
   previous_game = MagicMock()
   previous_game.find.return_value = previous_game_date
     
   current_game_date = MagicMock()
   current_game_date.text = 'January 7'
   current_game = MagicMock() 
   current_game.find.return_value = current_game_date
   
   games = [previous_game, current_game]
   
   index = 1
   year = 2024 
   expected_rest_days = date(2025, 1, 7) - date(2024, 12, 31)
   
   actual_rest_days = pfr_scraper.calculate_rest_days(games, index, year)
   
   assert expected_rest_days == actual_rest_days


@patch('scraping.pfr_scraper.get_game_date')
def test_remove_uneeded_games_removes_playoff_games(mocked_get_game_date):
   # set up game date mocks 
   mocked_game_one = MagicMock()
   mocked_game_two = MagicMock()
   
   mocked_game_date_two = MagicMock()
   mocked_game_date_two.text = 'Playoffs'
   mocked_game_two.find.return_value = mocked_game_date_two
   
   mocked_game_date_one = MagicMock()
   mocked_game_date_one.text = 'Random'
   mocked_game_one.find.return_value = mocked_game_date_one
   
   mocked_games = [mocked_game_one, mocked_game_two]
   expected_games = [mocked_game_one]
   
   mocked_get_game_date.return_value = date.today() - timedelta(days=1) # set date to be yesterday
   
   # act 
   pfr_scraper.remove_uneeded_games(mocked_games) 
   
   # assert 
   assert mocked_games == expected_games
   

@patch('scraping.pfr_scraper.get_game_date')
def test_remove_uneeded_games_removes_removes_bye_weeks(mocked_get_game_date):
      # set up game date mocks 
   mocked_game_one = MagicMock()
   mocked_game_two = MagicMock()
   
   mocked_game_date_two = MagicMock()
   mocked_game_date_two.text = 'Bye Week'
   mocked_game_two.find.return_value = mocked_game_date_two
   
   mocked_game_date_one = MagicMock()
   mocked_game_date_one.text = 'Random'
   mocked_game_one.find.return_value = mocked_game_date_one
   
   mocked_games = [mocked_game_one, mocked_game_two]
   expected_games = [mocked_game_one]
   
   mocked_get_game_date.return_value = date.today() - timedelta(days=1) # set date to be yesterday
   
   # act 
   pfr_scraper.remove_uneeded_games(mocked_games) 
   
   # assert 
   assert mocked_games == expected_games

@patch('scraping.pfr_scraper.get_game_date')
def test_remove_uneeded_games_removes_canceled_games(mocked_get_game_date):
   # set up game date mocks 
   mocked_game_one = MagicMock()
   mocked_game_two = MagicMock()
   
   mocked_game_date_two = MagicMock()
   mocked_game_date_two.text = 'canceled'
   mocked_game_two.find.return_value = mocked_game_date_two
   
   mocked_game_date_one = MagicMock()
   mocked_game_date_one.text = 'Random'
   mocked_game_one.find.return_value = mocked_game_date_one
   
   mocked_games = [mocked_game_one, mocked_game_two]
   expected_games = [mocked_game_one]
   
   mocked_get_game_date.return_value = date.today() - timedelta(days=1) # set date to be yesterday
   
   # act 
   pfr_scraper.remove_uneeded_games(mocked_games) 
   
   # assert 
   assert mocked_games == expected_games

@patch('scraping.pfr_scraper.get_game_date')
def test_remove_uneeded_games_removes_games_yet_to_be_played(mocked_get_game_date):
      # set up game date mocks 
   mocked_game_one = MagicMock()
   mocked_game_two = MagicMock()
   
   mocked_game_date_two = MagicMock()
   mocked_game_date_two.text = 'Random'
   mocked_game_two.find.return_value = mocked_game_date_two
   
   mocked_game_date_one = MagicMock()
   mocked_game_date_one.text = 'Random'
   mocked_game_one.find.return_value = mocked_game_date_one
   
   mocked_games = [mocked_game_one, mocked_game_two]
   expected_games = [mocked_game_one]
   
   mocked_get_game_date.return_value = date.today() + timedelta(days=1) # set date to be tomorrow, so it should be remove
   
   # act 
   pfr_scraper.remove_uneeded_games(mocked_games) 
   
   # assert 
   assert mocked_games == [] 
     