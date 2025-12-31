#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the lexical graph.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib

from icecream import ic
import networkx as nx
import numpy as np
import pandas as pd

from .elem import Entity, STRW_PREFIX
from .opt import calc_quantile_bins, stripe_column, root_mean_square


class LexicalGraph:
    """
Construct a _lexical graph_ from the parsed text using  a _textgraph_
algorithm and its distillation approach called _textrank_.
This graph gets used for ranking entities, then discarded later.
    """

    def __init__ (
        self,
        config: dict,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config
        self.lex_graph: nx.MultiDiGraph = nx.MultiDiGraph()


    def increment_edge (
        self,
        edge: tuple,
        ) -> None:
        """
Increment the incidence count for a relation, creating the edge first
if needed.
        """
        if not self.lex_graph.has_edge(*edge):
            # add a relation into the graph
            self.lex_graph.add_edge(
                edge[0],
                edge[1],
                key = edge[2],
                count = 1,
            )
        else:
            # relation already exists, increment count
            self.lex_graph[edge[0]][edge[1]][edge[2]]["count"] += 1


    def add_sent (
        self,
        ent_seq: list[ Entity ],
        *,
        debug: bool = False,
        ) -> None:
        """
Augment the _textgraph_ nodes and edges from the entities co-located
in a sentence.
        """
        sem_rel: str = f"{STRW_PREFIX}follows_lexically"

        if debug:
            ic(ent_seq)

        for hop in range(self.config["tr"]["tr_lookback"]):
            for ent_pos, ent in enumerate(ent_seq[: -1 - hop]):
                rel: Entity = ent_seq[hop + ent_pos + 1]
                edge: tuple = ( ent.uid, rel.uid, sem_rel, )

                if debug:
                    ic(edge)

                if not self.lex_graph.has_node(ent.uid):
                    self.lex_graph.add_node(
                        ent.uid,
                        lemma_key = ent.lemma_key,
                        count = ent.count,
                    )

                if not self.lex_graph.has_node(rel.uid):
                    self.lex_graph.add_node(
                        rel.uid,
                        lemma_key = rel.lemma_key,
                        count = rel.count,
                    )

                self.increment_edge(edge)


    def run_textrank (
        self,
        ) -> pd.DataFrame:
        """
Run eigenvector centrality (i.e., _Personalized PageRank_) to rank the
entities using an adapted `TextRank` algorithm implementation based 
on `NetworkX` and `Polars`.
        """
        # build a dataframe of node ranks and counts
        df_rank: pd.DataFrame = pd.DataFrame.from_dict([  # type: ignore
            {
                "node_id": node_id,
                "weight": rank,
                "count": self.lex_graph.nodes[node_id]["count"],
            }
            for node_id, rank in nx.pagerank(
                    self.lex_graph,
                    alpha = self.config["tr"]["tr_alpha"],
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

        return df_rank


    def load_graph (
        self,
        lex_path: pathlib.Path,
        ) -> None:
        """
De-serialize the lexical graph from a JSON file represented in the
_node-link_ data format.
        """
        with lex_path.open("r", encoding = "utf-8") as fp:
            self.lex_graph = nx.node_link_graph(
                json.load(fp),
                edges = "edges",
            )


    def save_graph (
        self,
        lex_path: pathlib.Path,
        ) -> None:
        """
Serialize the lexical graph as a JSON file represented in the
_node-link_ data format.

Aternatively this could be stored in a graph database.
        """
        with lex_path.open("w", encoding = "utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(
                        self.lex_graph,
                        edges = "edges",
                    ),
                    indent = 2,
                    sort_keys = True,
                )
            )
