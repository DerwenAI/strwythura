#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An adapted `TextRank` algorithm implementation based on `NetworkX` and `Polars`.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import defaultdict
import itertools
import typing

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

from .elem import Entity, NodeKind, StrwVocab
from .opt import calc_quantile_bins, stripe_column, root_mean_square


def run_textrank (
    config: dict,
    lex_graph: nx.MultiDiGraph,
    ) -> pl.DataFrame:
    """
Run eigenvalue centrality (i.e., _Personalized PageRank_) to rank the entities.
    """
    # build a dataframe of node ranks and counts
    df_rank: pd.DataFrame = pd.DataFrame.from_dict([  # type: ignore
        {
            "node_id": node,
            "weight": rank,
            "count": lex_graph.nodes[node]["count"],
        }
        for node, rank in nx.pagerank(
                lex_graph,
                alpha = config["tr"]["tr_alpha"],
                weight = "count",
        ).items()
    ])

    # normalize by column and calculate quantiles
    df1: pd.DataFrame = df_rank[[ "count", "weight" ]].apply(lambda x: x / x.max(), axis = 0)
    bins: np.ndarray = calc_quantile_bins(len(df1.index))

    # stripe each columns
    df2: pd.DataFrame = pd.DataFrame([
        stripe_column(values, bins)  # type: ignore
        for _, values in df1.items()
    ]).T

    # renormalize the ranks
    df_rank["rank"] = df2.apply(root_mean_square, axis = 1)
    rank_col: np.ndarray = df_rank["rank"].to_numpy()
    rank_col /= sum(rank_col)
    df_rank["rank"] = rank_col

    # move the ranked weights back into the graph
    for _, row in df_rank.iterrows():
        node: int = row["node_id"]
        lex_graph.nodes[node]["rank"] = row["rank"]

    df: pl.DataFrame = pl.DataFrame([
        node_attr
        for node, node_attr in lex_graph.nodes(data = True)
        if node_attr["kind"] == NodeKind.ENTITY.value
    ]).sort(
        [ "rank" ],
        descending = True,
    )

    return df


def cooccur_entities (
    lex_graph: nx.MultiDiGraph,
    span_decoder: typing.Dict[ tuple, Entity ],
    ) -> None:
    """
Connect entities which co-occur within the same sentence.
    """
    ent_map: typing.Dict[ int, typing.Set[ int ]] = defaultdict(set)

    for ent in span_decoder.values():
        if ent.node is not None:
            ent_map[ent.sent_id].add(ent.node)

    for nodes in ent_map.values():
        for pair in itertools.combinations(list(nodes), 2):
            if not lex_graph.has_edge(*pair):
                lex_graph.add_edge(
                    pair[0],
                    pair[1],
                    key = StrwVocab.CO_OCCURS_WITH.value,
                    prob = 1.0,
                )
