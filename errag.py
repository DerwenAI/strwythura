#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG example, based on DSPy.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import time
import traceback
import warnings

from icecream import ic
import dspy

from strwythura import DomainContext, GraphRAG, PerfProfiler, Strwythura


if __name__ == "__main__":
    # start the performance profiling
    profiler: PerfProfiler = PerfProfiler()
    profiler.start()

    # load the domain context
    domain: DomainContext = DomainContext()

    strw: Strwythura = Strwythura(
        domain,
    )

    # build a GraphRAG instance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        domain.load_assets()

        rag: GraphRAG = GraphRAG(strw)

        try:
            # loop to answer questions
            while True:
                question: str = input("\nQuoi? ").strip()

                if len(question) < 1:
                    continue

                if question.lower() in [ "quitter", "bye", "adieu" ]:
                    break

                response: dspy.primitives.prediction.Prediction = rag.qa_cycle(
                    question,
                    debug = True, # False
                )

                ic(question)
                ic(response.response)
                print("-" * 10)

        except EOFError:
            print("")
        except Exception as ex:
            ic(ex)
            traceback.print_exc()
        finally:
            print("\nÀ bientôt!\n")
            time.sleep(.1)

    # report the performance profiler stats
    profiler.report()
