#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strwythura: curate/iterate on the semantic layer.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import pathlib
import typing
import warnings

from icecream import ic
import polars as pl

from strwythura import Strwythura


def main ():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        strw: Strwythura = Strwythura()
        strw.load_assets()

    # construct a DataFrame as a view of the entities
    df: pl.DataFrame = pl.DataFrame([
        {
            "label": data["label"],
            "rank": round(float(data["rank"]), 4),
            "count": int(data["count"]),
            "text": data["text"].replace("\n", " "),
            "key": data["key"],
            "id": node,
        }
        for node, data in strw.sem_layer.nodes(data = True)
        if data["kind"] in [ "Entity", ]
    ]).sort(
        "label",
        "rank",
        "count",
        descending = [ True, False, False ],
    )

    ic(df.head())

    df.write_csv(
        pathlib.Path("data/sem.csv"),
        separator = ",",
    )


if __name__ == "__main__":
    main()
