from unittest.mock import patch, MagicMock
from scraping import pfr_scraper
import pandas as pd
import pytest
from scraping_helper import mock_find_common_metrics, mock_find_wr_metrics, \
         mock_find_rb_metrics, mock_find_qb_metrics, setup_game_log_mocks, \
         mock_add_common_game_log_metrics, mock_add_wr_game_log_metrics

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
   

