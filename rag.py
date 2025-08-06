#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG example,
based on BAML <https://docs.boundaryml.com/examples/prompt-engineering/retrieval-augmented-generation>

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import traceback
import warnings

from icecream import ic

from strwythura import Strwythura, GraphRAG
    

if __name__ == "__main__":
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
