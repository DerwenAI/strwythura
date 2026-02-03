#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Part 7: run enhanced GraphRAG, based on DSPy/Ollama/Opik, using the
assets developed in previous step.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib

from strwythura import GraphRAG, Profiler, Workflow


if __name__ == "__main__":
    # instantiate and configure our workflow manager
    work: Workflow = Workflow(
        config_path = pathlib.Path("config.toml"),
    )

    # start the performance profiling
    profiling: bool = work.config["prof"]["use_pyinst"]

    if profiling:
        prof: Profiler = Profiler()

    # finish provisioning
    work.load_assets()
    work.ctx.open_vector_tables()
    work.load_parser()

    domain: dict = json.load(
        pathlib.Path("domain.json").open("r", encoding = "utf-8")
    )

    # leverage the workflow assets for this domain for a use case:
    # run enhanced GraphRAG for a question/answer chat bot
    rag: GraphRAG = GraphRAG(
        work,
        domain["name"],
        domain["description"],
        run_local = True,
        use_opik = True,
    )

    rag.question_answer(
        debug = False, # True
    )


    # finally, report the performance profiler stats
    if profiling:
        prof.analyze()
