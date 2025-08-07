#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG example,
based on BAML <https://docs.boundaryml.com/examples/prompt-engineering/retrieval-augmented-generation>

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import time
import traceback
import warnings

from icecream import ic
import baml_py

from strwythura import Strwythura, GraphRAG
from strwythura.baml_client import types as baml_types


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # build a GraphRAG instance
        strw: Strwythura = Strwythura()
        strw.load_assets()

        rag: GraphRAG = GraphRAG(strw)

        try:
            # loop to answer questions
            while True:
                question: str = input("\nQuoi? ").strip()

                if question.lower() in [ "quitter", "bye" ]:
                    break

                assert len(question) > 0

                response: baml_types.Response = rag.qa_cycle(
                    question,
                    debug = False, # True
                )

                ic(question)
                ic(response)
                print("-" * 10)

        except EOFError:
            print("")
        except baml_py.internal_monkeypatch.BamlValidationError as baml_ex:
            ic(ex)
        except Exception as ex:
            ic(ex)
            traceback.print_exc()
        finally:
            print("\nÀ bientôt!\n")
            time.sleep(.1)
