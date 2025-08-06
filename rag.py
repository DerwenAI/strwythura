#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG example,
based on BAML <https://docs.boundaryml.com/examples/prompt-engineering/retrieval-augmented-generation>

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import logging
import os
import traceback
import warnings

from icecream import ic

from strwythura import Strwythura, GraphRAG
    

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

        try:
            # loop to answer questions
            rag.qa_loop(debug = False) # True

        except EOFError:
            print("")
        except Exception as ex:
            ic(ex)
            traceback.print_exc()
