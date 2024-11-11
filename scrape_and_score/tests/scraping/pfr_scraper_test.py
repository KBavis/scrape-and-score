from unittest.mock import patch, MagicMock
from scraping import pfr_scraper
import pytest
from helper import mock_find_common_metrics, mock_find_wr_metrics, mock_find_rb_metrics, mock_find_qb_metrics

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