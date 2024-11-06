from unittest.mock import patch, MagicMock
from scraping import pfr_scraper
import pytest

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
   
   tr_mock = MagicMock()
   tr_mock.find.return_value = 10