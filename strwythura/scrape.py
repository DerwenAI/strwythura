#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collect unstructured data from specific web page sources.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing

from bs4 import BeautifulSoup
import requests
import spacy

from .graph import TextChunk
from .nlp import make_chunk


SCRAPE_HEADERS: typing.Dict[ str, str ] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
}


def scrape_html (
    simple_pipe: spacy.Language,
    url: str,
    chunk_list: typing.List[ TextChunk ],
    chunk_id: int,
    ) -> int:
    """
A simple web page text scraper, which also performs chunking.
Returns the updated `chunk_id` index.
    """
    response: requests.Response = requests.get(
        url,
        headers = SCRAPE_HEADERS,
    )

    soup: BeautifulSoup = BeautifulSoup(
        response.text,
        features = "lxml",
    )

    scrape_doc: spacy.tokens.doc.Doc = simple_pipe("\n".join([
        para.text.strip()
        for para in soup.find_all("p")
    ]))

    chunk_id = make_chunk(
        scrape_doc,
        url,
        chunk_list,
        chunk_id,
    )

    return chunk_id
