#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG example,
based on BAML <https://docs.boundaryml.com/examples/prompt-engineering/retrieval-augmented-generation>

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import logging
import os
import pathlib
import sys
import traceback
import typing
import warnings

from icecream import ic
import baml_client
import gensim
import lancedb
import loguru
import networkx as nx
import spacy

import gliner_spacy.pipeline
import glirel


from strwythura import GraphRAG, \
    CHUNK_TABLE, KG_PATH, LANCEDB_URI, W2V_PATH, \
    SPACY_MODEL, RE_LABELS, init_nlp_pipe


def extract_entities (
    nlp_pipe: spacy.Language,
    question: str,
    ) -> typing.Iterator[ str ]:
    """
Extract entity spans from a text question.
    """
    doc: spacy.tokens.doc.Doc = list(
        nlp_pipe.pipe(
            [( question, RE_LABELS )],
            as_tuples = True,
        )
    )[0][0]

    for span in doc.ents:
        key: str = " ".join([
            tok.pos_ + "." + tok.lemma_.strip().lower()
            for tok in span
        ])
        
        yield key
    

def qa_loop (
    rag: GraphRAG,
    nlp_pipe: spacy.Language,
    ) -> None:
    """
Loop to answer questions.
    """
    while True:
        question: str = input("\n\nWhat is your question? ").strip()

        if question.lower() in [ "exit", "quit" ]:
            print("\nÀ bientôt!")
            break

        entities: typing.List[ str ] = list(extract_entities(nlp_pipe, question))
        chunks: typing.List[ str ] = rag.get_chunks(question, entities, debug = False) # True
        context: str = "\n".join( chunks )
        response: baml_client.types.Response = baml_client.b.RAG(question, context)

        ic(response)
        print("-" * 10)
    

if __name__ == "__main__":
    # configuration
    os.environ["BAML_LOG"] = "WARN"

    # none of this works!
    #os.environ["TQDM_DISABLE"] = "1"
    #loguru.logger.disable(gliner_spacy.pipeline.__name__)
    #loggers: dict = { name:logging.getLogger(name) for name in logging.root.manager.loggerDict }
    #ic(loggers)
    #logging.getLogger("glirel.spacy_integration").setLevel(logging.ERROR)

    # load the serialized assets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nlp_pipe: spacy.Language = init_nlp_pipe()

    w2v_model: gensim.models.Word2Vec = gensim.models.Word2Vec.load(W2V_PATH)

    sem_overlay: nx.Graph = nx.Graph()

    with pathlib.Path(KG_PATH).open("r", encoding = "utf-8") as fp:
        sem_overlay = nx.node_link_graph(
            json.load(fp),
            edges = "links",
        )

    vect_db: lancedb.db.LanceDBConnection = lancedb.connect(LANCEDB_URI)
    chunk_table: lancedb.table.LanceTable = vect_db.open_table(CHUNK_TABLE)

    # build a GraphRAG instance
    rag: GraphRAG = GraphRAG(
        chunk_table,
        w2v_model,
        sem_overlay,
    )

    # loop to answer questions
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qa_loop(rag, nlp_pipe)

    except EOFError:
        print("")
        pass
    except Exception as ex:
        ic(ex)
        traceback.print_exc()
