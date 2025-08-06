#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Use the generated assets to run GraphRAG
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing

from icecream import ic
import gensim
import lancedb
import networkx as nx
import pandas as pd
import spacy


class GraphRAG:
    """
Run an example query through LanceDB to identify _chunks_ and through
the Word2Vec entity embedding model for a _semantic expansion_ to
produce a set of _anchor nodes_ in the NetworkX graph.
    """

    def __init__ (
        self,
        chunk_table: lancedb.table.LanceTable,
        w2v_model: gensim.models.Word2Vec,
        sem_overlay: nx.Graph,
        ) -> None:
        """
Constructor.
        """
        self.chunk_table: lancedb.table.LanceTable = chunk_table
        self.w2v_model: gensim.models.Word2Vec = w2v_model
        self.sem_overlay: nx.Graph = sem_overlay


    def get_chunks (
        self,
        query: str,
        entities: typing.List[ str ],
        *,
        debug: bool = False,
        num_chunks: int = 10,
        ) -> typing.List[ str ]:
        """
Run semantic search to produce a set of text chunks.
        """
        # show the query
        if debug:
            ic(query)

        # enumerate chunks from a vector search -- the basic RAG process
        df_chunk: pd.DataFrame = self.chunk_table.search(query).to_pandas()

        if debug:
            ic(df_chunk)

            for row in df_chunk.itertuples():
                ic(row.text)

        # enumerate neighbor entities from entity embedding
        neighbors: list = []
        
        for entity in entities:
            try:
                neighbor_iter = self.w2v_model.wv.most_similar(
                    positive = [ entity ],
                    topn = num_chunks,
                )

                for neighbor in neighbor_iter:
                    neighbors.append(neighbor)
            except KeyError:
                pass

        df_entity: pd.DataFrame = pd.DataFrame([
            {
                "entity": neighbor[0],
                "distance": neighbor[1],
            }
            for neighbor in neighbors
            if neighbor[1] > 0.0
        ])

        if debug:
            ic(df_entity)

        # perform a semantic expansion to enrich the anchor nodes
        if len(df_entity) > 0:
            expansion: typing.Set[ str ] = set(df_entity["entity"].values.tolist())

            for node, dat in self.sem_overlay.nodes(data = True):
                if "key" in dat and dat["key"] in expansion:
                    if debug:
                        ic(node, dat)

        # return a list of text chunks
        return [
            row.text
            for row in df_chunk.itertuples()
        ]
