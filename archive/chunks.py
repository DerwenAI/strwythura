#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test rig"""

import pathlib
import tomllib
import typing
import unicodedata

from icecream import ic
from strwythura import Scraper, TextChunk


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
])


def scrub_text (
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
        swap: str | None = CHAR_MAP.get(rach)

        if swap is not None:
            exploded.append(swap)
        elif rach not in SKIP_CHARS:
            exploded.append(char)

    min_scrub: str = unicodedata.normalize(
        "NFKD",
        "".join(exploded),
    ).replace("\u200b", "").strip()

    #max_scrub: str = min_scrub.encode("ascii", "ignore").decode("utf-8").strip()

    return min_scrub


CHUNK_SIZE: int = 1000

def make_chunks (
    chunks: list[ str ],
    ) -> typing.Iterator[ list[ str ] ]:
    """
Iterate through the paragraphs parsed from an article, assembling its
text chunks.
    """
    sum_chars: int = 0
    bucket: list[ str ] = []

    for text in chunks:
        text = scrub_text(text)
        num_chars: int = len(text)

        if num_chars > 0:
            ic(num_chars, text[:11].strip())

            if (sum_chars + num_chars) < CHUNK_SIZE:
                bucket.append(text)
                sum_chars += num_chars
            else:
                # emit prev bucket
                yield bucket
                bucket = [ text ]
                sum_chars = num_chars

    # emit last bucket
    yield bucket


if __name__ == "__main__":
    config_path: pathlib.Path = pathlib.Path("config.toml")

    with config_path.open("rb") as fp:
        config: dict = tomllib.load(fp)
        scraper: Scraper = Scraper(config)

    url: str = "https://aaic.alz.org/releases-2024/processed-red-meat-raises-risk-of-dementia.asp"

    for texts in make_chunks(scraper.scrape_html(url)):
        text: str = "\n\n".join(texts)

        chunk: TextChunk = TextChunk(
            uid = 0,
            url = url,
            sent_id = 1,
            text = text,
        )

        ic(chunk)
