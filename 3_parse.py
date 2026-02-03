#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Part 3: crawl and parse unstructured content, initializing the vector
store and generating a lexical graph.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib

from strwythura import Profiler, Workflow


if __name__ == "__main__":
    # instantiate and configure our workflow manager
    work: Workflow = Workflow(
        config_path = pathlib.Path("config.toml"),
    )

    # start the performance profiling
    profiling: bool = work.config["prof"]["use_pyinst"]

    if profiling:
        prof: Profiler = Profiler()

    # de-serialize assets from previous steps
    work.thesaurus.load_source(
        pathlib.Path(work.config["sz"]["thesaurus_path"]),
        format = "turtle",
    )

    work.ctx.ent_store.load_json(
        pathlib.Path(work.config["ent"]["store_path"]),
    )

    work.ctx.erkg.load_graph(
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )

    work.ctx.open_vector_tables()
    work.load_parser()

    domain: dict = json.load(
        pathlib.Path("domain.json").open("r", encoding = "utf-8")
    )

    # crawl, chunk, parse -- across all the documents
    work.crawl_chunk_parse(
        domain["sources"]["content"],
    )

    # serialize the intermediate results
    work.ctx.ent_store.save_json(
        pathlib.Path(work.config["ent"]["store_path"]),
    )

    work.ctx.ent_store.save_vec(
        pathlib.Path(work.config["ent"]["vec_path"]),
    )

    work.ctx.lex.save_graph(
        pathlib.Path(work.config["nlp"]["lex_path"]),
    )

    work.ctx.erkg.save_graph(
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )


    # finally, report the performance profiler stats
    if profiling:
        prof.analyze()
