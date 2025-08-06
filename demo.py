#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphGeeks.org talk 2024-08-14 https://live.zoho.com/PBOB6fvr6c
How to construct _knowledge graphs_ from unstructured data sources.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib
import traceback
import tracemalloc
import typing
import warnings

from icecream import ic
from pyinstrument import Profiler
import gensim
import lancedb
import networkx as nx
import spacy

from strwythura import CHUNK_TABLE, HTML_PATH, KG_PATH, LANCEDB_URI, W2V_PATH, SPACY_MODEL, \
    TextChunk, GraphRAG, \
    construct_kg, \
    train_entity_model, \
    gen_pyvis


def main (
    debug: bool = False,
    ) -> int:
    """
Main entry point.
    """
    # define the global data structures
    url_list: typing.List[ str ] = [
        "https://aaic.alz.org/releases-2024/processed-red-meat-raises-risk-of-dementia.asp",
        "https://www.theguardian.com/society/article/2024/jul/31/eating-processed-red-meat-could-increase-risk-of-dementia-study-finds",
    ]

    simple_pipe: spacy.Language = spacy.load(SPACY_MODEL)

    vect_db: lancedb.db.LanceDBConnection = lancedb.connect(LANCEDB_URI)

    chunk_table: lancedb.table.LanceTable = vect_db.create_table(
        CHUNK_TABLE,
        schema = TextChunk,
        mode = "overwrite",
    )

    sem_overlay: nx.Graph = nx.Graph()

    try:
        w2v_vectors: list = []

        construct_kg(
            url_list,
            simple_pipe,
            chunk_table,
            sem_overlay,
            w2v_vectors,
            debug = debug,
        )

        # serialize the resulting KG
        with pathlib.Path(KG_PATH).open("w", encoding = "utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(sem_overlay, edges = "links"),
                    indent = 2,
                    sort_keys = True,
                )
            )

        # generate HTML for an interactive visualization of a graph
        gen_pyvis(
            sem_overlay,
            HTML_PATH,
            num_docs = len(url_list),
        )

        # train an entity embedding model
        w2v_model: gensim.models.Word2Vec = train_entity_model(
            w2v_vectors,
            W2V_PATH,
            debug = debug,
        )


        ######################################################################
        # prepare chunk list from an example query

        query: str = "cognitive decline"

        entity: str = " ".join([
            f"{token.pos_}.{token.lemma_}"
            for token in simple_pipe(query)
        ])

        rag: GraphRAG = GraphRAG(
            chunk_table,
            w2v_model,
            sem_overlay,
            )

        rag.get_chunks(
            query,
            [ entity ],
            debug = debug,
        )
    except Exception as ex:
        ic(ex)
        traceback.print_exc()


######################################################################
## main entry point

if __name__ == "__main__":
    # start the stochastic call trace profiler and memory profiler
    profiler: Profiler = Profiler()
    profiler.start()
    tracemalloc.start()

    # construct the KG and build the GraphRAG assets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(debug = True)

    # stop the profiler and report performance statistics
    profiler.stop()
    profiler.print()

    # report the memory usage
    report: tuple = tracemalloc.get_traced_memory()
    peak: float = round(report[1] / 1024.0 / 1024.0, 2)
    print(f"peak memory usage: {peak} MB")
