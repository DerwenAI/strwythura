#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple analysis of semantics

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib
import tomllib
import typing

from icecream import ic
import networkx as nx
import pandas as pd


if __name__ == "__main__":
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

        node_list: typing.List[ dict ] = []

        for node, data in sem_overlay.nodes(data = True):
            if data["kind"] in [ "Entity", ]:
                data["id"] = node
                node_list.append(data)

        df: pd.DataFrame = pd.DataFrame(node_list)
        col = df.pop("id")
        df.insert(0, "id", col)

        ic(df.head())
