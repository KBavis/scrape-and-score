import pytest
import requests
from unittest.mock import MagicMock
import builtins
from io import StringIO
import json
from datetime import datetime, timedelta

"""
Module utilized to hold helper functionality regarding unit tests for our proxy package, 
including our constants and some mocked functionality
"""

# constants
URL = "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=http&proxy_format=ipport&format=json"
IP = "141.98.153.86:80"
IP_ADDRESSES = ["141.98.153.86:80", "157.254.53.50:80", "47.91.104.88:3128"]
JSON_RESPONSE = []
EXPECTED_PROXIES = []
for ip_address in IP_ADDRESSES:
    EXPECTED_PROXIES.append({"http": ip_address})
    JSON_RESPONSE.append({"proxy": ip_address})


# get mocked html IP
def get_html_ip():
    html_element = MagicMock()
    html_element.text = IP
    html_element.raise_for_status = MagicMock()
    return html_element


# mock for requests.get() to return expected list of proxies
@pytest.fixture
def mock_proxies_response(monkeypatch):

    # mock "requests.Response"
    class MockResponse:
        # mock "requests.get().json()"
        @staticmethod
        def json():
            return {"proxies": JSON_RESPONSE}

    # mock "requests.get()"
    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)


# mock for 'with open()' to return expected file
@pytest.fixture
def mock_with_open(monkeypatch):
    expiration_time = datetime.now() + timedelta(hours=3)
    mock_proxies = {
        "proxies": JSON_RESPONSE,
        "expiration": expiration_time.isoformat(),
    }  # mock proxies

    # create mock file to be opened by "with open()"
    mock_file = StringIO(json.dumps(mock_proxies))

    # mock "with open()"
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: mock_file)


# mock for 'with open()' to return expected file
@pytest.fixture
def mock_with_open_expired_cache(monkeypatch):
    expiration_time = datetime.now() - timedelta(hours=3)
    mock_proxies = {
        "proxies": JSON_RESPONSE,
        "expiration": expiration_time.isoformat(),
    }  # mock proxies

    # create mock file to be opened by "with open()"
    mock_file = StringIO(json.dumps(mock_proxies))

    # mock "with open()"
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: mock_file)
