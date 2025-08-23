#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strwythura: curate/iterate on the semantic layer.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib
import tomllib
import typing

from icecream import ic
import networkx as nx
import polars as pl


def main ():
    # load the graph data
    with open(pathlib.Path("config.toml"), mode = "rb") as fp:
        config: dict = tomllib.load(fp)

    sem_path: pathlib.Path = pathlib.Path(config["kg"]["sem_path"])

    with pathlib.Path(sem_path).open("r", encoding = "utf-8") as fp:
        ner_labels: typing.List[ str ] = json.load(fp)

    kg_path: pathlib.Path = pathlib.Path(config["kg"]["kg_path"])

    with pathlib.Path(kg_path).open("r", encoding = "utf-8") as fp:
        sem_overlay: nx.Graph = nx.node_link_graph(
            json.load(fp),
            edges = "edges",
        )

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
        for node, data in sem_overlay.nodes(data = True)
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
