import requests
import pytest
import json
import os
from unittest.mock import Mock
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
@pytest.mark.usefixtures("mock_with_open")
def test_cache_proxies():
    proxy.cache_proxies(EXPECTED_PROXIES)

    # ensure data is cached
    file = "./scrape_and_score/resources/proxies.json"
    assert os.path.exists(file)

# validate functions called when no cache
def test_get_proxy_calls_fetch_proxies_no_cache():
    # create mock callback
    mock_callback = Mock() 

    proxy.get_proxy(mock_callback)

    

# def test_fetch_proxies_validates_each_proxy(): 

# def test_get_proxy_calls_fetch_proxies_if_none_cached():

# def test_get_proxy_calls_fetch_proxies_if_cache_expired():

# def test_get_proxy_calls_get_random_proxy(): 

