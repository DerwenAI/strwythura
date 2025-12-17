#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collect unstructured content (text) from specific web page sources.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import ssl
import unicodedata

from bs4 import BeautifulSoup
import requests
import requests_cache


class Scraper:
    """
A simple HTML scraper which caches content.
    """
    SCRAPE_HEADERS: dict[ str, str ] = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",  # pylint: disable=C0301
    }

    CHAR_MAP: dict[ int, str ] = {
        226: "'",
        8220: '"',
        8221: '"',
        8216: "'",
        8217: "'",
    }

    SKIP_CHARS: set[ int ] = set([
        128,
        148,
        153,
        156,
        157,
        8203,
    ])


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


    def scrub_text (
        self,
        text: str | None,
        ) -> str | None:
        """
Scrub text of non-printable characters, typesetting artifacts, UTF-8 errors, etc.
Courtesy of <https://github.com/DerwenAI/pytextrank>
        """
        if text is None:
            return None

        # explode the string into characters to be mapped
        exploded = []

        for char in text:
            rach: int = ord(char)
            swap: str | None = self.CHAR_MAP.get(rach)

            if swap is not None:
                exploded.append(swap)
            elif rach not in self.SKIP_CHARS:
                exploded.append(char)

        min_scrub: str = unicodedata.normalize(
            "NFKD",
            "".join(exploded),
        ).strip()

        #max_scrub: str = min_scrub.encode("ascii", "ignore").decode("utf-8").strip()

        return min_scrub


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
            headers = self.SCRAPE_HEADERS,
        )

        soup: BeautifulSoup = BeautifulSoup(
            response.text,
            features = "lxml",
        )

        return [
            self.scrub_text(para.text.strip())  # type: ignore
            for para in soup.find_all("p")
        ]
