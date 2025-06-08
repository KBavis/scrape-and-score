import requests
import logging
import time
from scrape_and_score.proxy import proxy
from scrape_and_score.config import get_config

session = requests.Session()


def fetch_page(url: str):
    """
    Functionality to fetch the raw HTML corresponding to a particular URL

    Args:
        url(str): the URL we want to extract the raw HTML from

    Returns:
        str: raw HTML corresponding to specified URL
    """

    try:
        logging.info(f"Fetching raw HTML from the following URL: {url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
            "Connection": "keep-alive",
        }

        time.sleep(get_config("scraping.delay"))

        response = session.get(url, proxies=proxy.get_proxy(), headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error while fetching HTML content from the following URL {url} : {e}")
        return None
