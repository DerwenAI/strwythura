#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collect unstructured content (text) from specific web page sources.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import ssl

from bs4 import BeautifulSoup
import requests
import requests_cache
import spacy


SCRAPE_HEADERS: dict[ str, str ] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",  # pylint: disable=C0301
}


class Scraper:
    """
A simple HTML scraper which caches content.
    """

    def __init__ (
        self,
        config: dict,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config
        self.session: requests_cache.CachedSession = self.get_cache()


    def get_cache (
        self,
        ) -> requests_cache.CachedSession:
        """
Build a URL request cache session, optionally loading any
previous serialized cache from disk.
        """
        session: requests_cache.CachedSession = requests_cache.CachedSession(
            backend = requests_cache.SQLiteCache(
                self.config["scraper"]["cache_path"],
            ),
        )

        session.settings.expire_after = self.config["scraper"]["cache_expire"]

        return session


    def scrape_html (
        self,
        url: str,
        ) -> list[ str ]:
        """
A simple web page content scraper, which returns a list of text paragraphs.
        """
        response: requests.Response = self.session.get(
            url,
            verify = ssl.CERT_NONE,
            timeout = 10,
            allow_redirects = True,
            headers = SCRAPE_HEADERS,
        )

        soup: BeautifulSoup = BeautifulSoup(
            response.text,
            features = "lxml",
        )

        return [
            para.text.strip()
            for para in soup.find_all("p")
        ]
