#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG example,
based on BAML <https://docs.boundaryml.com/examples/prompt-engineering/retrieval-augmented-generation>

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import os
import pathlib
import sys
import typing
import warnings

from icecream import ic
import baml_client
import gensim
import lancedb
import networkx as nx
import spacy

from strwythura import GraphRAG, \
    KG_PATH, LANCEDB_URI, SPACY_MODEL, W2V_PATH


if __name__ == "__main__":
    # configuration
    os.environ["BAML_LOG"] = "WARN"

    # load the serialized assets
    simple_pipe: spacy.Language = spacy.load(SPACY_MODEL)
    w2v_model: gensim.models.Word2Vec = gensim.models.Word2Vec.load(W2V_PATH)

    sem_overlay: nx.Graph = nx.Graph()

    with pathlib.Path(KG_PATH).open("r", encoding = "utf-8") as fp:
        sem_overlay = nx.node_link_graph(
            json.load(fp),
            edges = "links",
        )

    vect_db: lancedb.db.LanceDBConnection = lancedb.connect(LANCEDB_URI)
    chunk_table: lancedb.table.LanceTable = vect_db.open_table("chunk")

    # build a GraphRAG instance
    rag: GraphRAG = GraphRAG(
        simple_pipe,
        chunk_table,
        w2v_model,
        sem_overlay,
    )

    # loop to answer questions
    try:
        while True:
            question: str = input("\n\nWhat is your question? ").strip()

            if question.lower() in [ "exit", "quit" ]:
                print("\nÀ bientôt!")
                break

            query: str = "processed red meat"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chunks: typing.List[ str ] = rag.get_chunks(query)

            context: str = "\n".join( chunks )
            response: baml_client.types.Response = baml_client.b.RAG(question, context)

            ic(response)
            print("-" * 10)

    except EOFError:
        print("")
        pass
    except Exception as ex:
        ic(ex)
        traceback.print_exc()
