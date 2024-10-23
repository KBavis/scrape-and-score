import requests
import pytest
from datetime import datetime, timedelta
import json
import os
from unittest import mock
from scrape_and_score.proxy import proxy
from helper import mock_proxies_response, mock_with_open_expired_cache, get_html_ip, mock_with_open, EXPECTED_PROXIES, URL, IP, JSON_RESPONSE

# validate expected proxies returned when valid
@pytest.mark.usefixtures("mock_proxies_response")
def test_fetch_proxies_returns_expected_proxies(monkeypatch):
    monkeypatch.setattr(proxy, "validate_proxy", lambda proxy: True) 
    monkeypatch.setattr(proxy, "cache_proxies", lambda proxy: None)

    proxies = proxy.fetch_proxies(URL)
    assert proxies == EXPECTED_PROXIES

# validate no proxies returned when invalid
@pytest.mark.usefixtures("mock_proxies_response")
def test_fetch_proxies_returns_empty_list(monkeypatch):
    monkeypatch.setattr(proxy, "validate_proxy", lambda proxy: False)

    proxies = proxy.fetch_proxies(URL)
    assert len(proxies) == 0

# validate true is returned when proxy is valid
def test_validate_proxy_is_valid(monkeypatch):
    html_ip = get_html_ip()
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: html_ip)
    valid = proxy.validate_proxy(proxy)
    assert valid is True

# validate false is returned when proxy is invalid
def test_validate_proxy_is_invalid(monkeypatch):
    html_ip = get_html_ip()
    html_ip.text = "141.98.86:80" # update IP to be invalid
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: html_ip)
    valid = proxy.validate_proxy(proxy)
    assert valid is False

# validate false is returned when validating proxy causes exception 
def test_validate_proxy_is_invalid_when_exception(monkeypatch):
    html_ip = get_html_ip()
    html_ip.raise_for_status.side_effect =  Exception("Error") # mock exception to occur
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: html_ip)
    valid = proxy.validate_proxy(proxy)
    assert valid is False

# validate get_random_proxy() returns random proxy from cache
@pytest.mark.usefixtures("mock_with_open")
def test_get_random_proxy_returns_random_proxy():
    random_proxy = proxy.get_random_proxy()
    assert random_proxy in JSON_RESPONSE

# validate ability to determine cache isn't expired
@pytest.mark.usefixtures("mock_with_open")
def test_is_cache_expired_is_false():
    assert proxy.is_cache_expired() is False

# validate ability to determine cache is expired
@pytest.mark.usefixtures("mock_with_open_expired_cache")
def test_is_cache_expired_is_true():
    assert proxy.is_cache_expired() is True

# validate cache proxies correctly creates cache
def test_cache_proxies_creates_cache(tmpdir):
    # create temporary dir & cache 
    temp_file = tmpdir.join("proxies.json")
    
    # invoke function
    proxy.cache_proxies(EXPECTED_PROXIES, file_path=str(temp_file))

    # ensure data is cached
    assert os.path.exists(temp_file)

def test_cache_proxies_persists_proxies(tmpdir):
    # create tmp dir & cache 
    temp_file = tmpdir.join("proxies.json")
    
    # invoke function 
    proxy.cache_proxies(EXPECTED_PROXIES, file_path=str(temp_file))
    
    # validate proxies persisted 
    with open(temp_file, "r") as f:
        data = json.load(f)
        assert data["proxies"] == EXPECTED_PROXIES    
        
def test_cache_proxies_persists_expriation(tmpdir):    
    # create tmp dir & cache 
    temp_file = tmpdir.join("proxies.json")
    
    current_time = datetime.now()
    
    # invoke function 
    proxy.cache_proxies(EXPECTED_PROXIES, file_path=str(temp_file))
    
    # extract expiration persisted 
    with open(temp_file, "r") as f:
        data = json.load(f)
        expiration = datetime.fromisoformat(data["expiration"])
        
    time_diff = expiration - current_time
    
    # assert that expiration is three hours from now 
    assert time_diff - timedelta(hours=3) < timedelta(seconds=3)
        
        
def test_get_proxy_no_cache():
    # define mocks 
    with mock.patch("os.path.exists") as mock_exists, \
         mock.patch("scrape_and_score.proxy.proxy.fetch_proxies") as mock_fetch_proxies, \
         mock.patch("scrape_and_score.proxy.proxy.is_cache_expired") as mock_is_cache_expired, \
         mock.patch("scrape_and_score.proxy.proxy.get_random_proxy") as mock_get_random_proxy:     
            
         mock_exists.return_value = False # ensure cache DNE
         mock_fetch_proxies.return_value = EXPECTED_PROXIES
         mock_is_cache_expired.return_value = False 
         mock_get_random_proxy.return_value = EXPECTED_PROXIES[0]  
         
         # invoke function 
         proxy.get_proxy()              
         
         # verify mock interactions 
         mock_fetch_proxies.assert_called_once() 
         mock_get_random_proxy.assert_called_once()
         mock_exists.assert_called_once()
         mock_is_cache_expired.assert_called_once()
    
def test_get_proxy_cache_expired(): 
    # define mocks 
    with mock.patch("os.path.exists") as mock_exists, \
         mock.patch("scrape_and_score.proxy.proxy.fetch_proxies") as mock_fetch_proxies, \
         mock.patch("scrape_and_score.proxy.proxy.is_cache_expired") as mock_is_cache_expired, \
         mock.patch("scrape_and_score.proxy.proxy.get_random_proxy") as mock_get_random_proxy:     
            
         mock_exists.return_value = True # ensure cache exists 
         mock_fetch_proxies.return_value = EXPECTED_PROXIES
         mock_is_cache_expired.return_value = True # ensure cache expired 
         mock_get_random_proxy.return_value = EXPECTED_PROXIES[0]  
         
         # invoke function 
         proxy.get_proxy()              
         
         # verify mock interactions 
         mock_fetch_proxies.assert_called_once() 
         mock_get_random_proxy.assert_called_once()
         mock_exists.assert_called_once()
         mock_is_cache_expired.assert_called_once()

