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
import traceback
import typing
import warnings

from icecream import ic

from strwythura import Strwythura, GraphRAG
from strwythura import baml_client
 

def qa_loop (
    rag: GraphRAG,
    ) -> None:
    """
Loop to answer questions.
    """
    while True:
        question: str = input("\n\nWhat is your question? ").strip()

        if question.lower() in [ "exit", "quit" ]:
            print("\nÀ bientôt!")
            break

        chunks: typing.List[ str ] = rag.get_chunks(question, debug = False) # True
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # build a GraphRAG instance
        strw: Strwythura = Strwythura()
        strw.load_assets()

        rag: GraphRAG = GraphRAG(strw)

        # loop to answer questions
        try:
            qa_loop(rag)

        except EOFError:
            print("")
        except Exception as ex:
            ic(ex)
            traceback.print_exc()
