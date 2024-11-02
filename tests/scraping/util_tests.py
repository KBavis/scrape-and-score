import unittest
from unittest.mock import patch, MagicMock 
import requests 
from scrape_and_score.scraping import util


@patch('scrape_and_score.scraping.util.get_proxy')
@patch('scrape_and_score.scraping.util.session.get')
@patch('scrape_and_score.scraping.util.time.sleep', return_value=None)
def test_fetch_page_success(mock_sleep, mock_get, mock_get_proxy):
   # arrange 
   mock_get_proxy.return_value = {"http": "97.74.87.226:80"}
   
   mock_response = MagicMock() 
   mock_response.status_code = 200
   mock_response.text = "<html><body>This is a test</body></html>"
   mock_get.return_value = mock_response 
   
   url = "https://example.com"
   
   # act 
   result = util.fetch_page(url)
   
   # assert 
   assert result == mock_response.text

   
   