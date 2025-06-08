from scrape_and_score.scraping.scraping_util import fetch_page
import pytest
from unittest.mock import patch, MagicMock
import requests


@pytest.fixture
def url():
    return "https://example.com"


@patch("scrape_and_score.scraping.scraping_util.session.get")
@patch("scrape_and_score.scraping.scraping_util.time.sleep", return_value=None)
@patch("scrape_and_score.scraping.scraping_util.get_config", return_value=0)
@patch("scrape_and_score.scraping.scraping_util.proxy.get_proxy", return_value=None)
def test_fetch_page_success(mock_get_proxy, mock_get_config, mock_sleep, mock_get, url):
    # arrange mocks
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.text = "<html>some content</html>"
    mock_get.return_value = mock_response

    result = fetch_page(url)

    # validate HTML returned
    assert result == "<html>some content</html>"

    # validate expected function interactions
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == url
    assert "headers" in kwargs
    assert "proxies" in kwargs
    mock_sleep.assert_called_once_with(0)


@patch("scrape_and_score.scraping.scraping_util.session.get")
@patch("scrape_and_score.scraping.scraping_util.time.sleep", return_value=None)
@patch("scrape_and_score.scraping.scraping_util.get_config", return_value=0)
@patch("scrape_and_score.scraping.scraping_util.proxy.get_proxy", return_value=None)
def test_fetch_page_request_exception(
    mock_get_proxy, mock_get_config, mock_sleep, mock_get, url
):
    mock_get.side_effect = requests.RequestException("Network failure")

    result = fetch_page(url)

    assert result is None
    mock_get.assert_called_once()
