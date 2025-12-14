#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Part 2: populate a semantic layer from the entity resolution results
plus a domain taxonomy.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib

from strwythura import Profiler, Workflow


if __name__ == "__main__":
    # start the performance profiling -
    profiling: bool = True # False

    if profiling:
        prof: Profiler = Profiler()


    # instantiate and configure our workflow manager
    work: Workflow = Workflow(
        config_path = pathlib.Path("config.toml"),
    )

    work.ctx.open_vector_tables(create = True)
    work.load_parser()

    domain: dict = json.load(
        pathlib.Path("domain.json").open("r", encoding = "utf-8")
    )

    # populate a semantic layer from the entity resolution results
    work.populate_semantic_layer(
        pathlib.Path(work.config["sz"]["er_path"]),
        pathlib.Path(domain["taxonomy"]),
        language = domain["language"],
    )

    # promote semantic graph elements in `RDFlib` as a "backbone"
    # for an ERKG managed as a property graph in `NetworkX`
    work.build_graph_backbone()

    # serialize the intermediate results
    work.thesaurus.save_source(
        pathlib.Path(work.config["sz"]["thesaurus_path"]),
        format = "turtle",
    )

    work.ctx.ent_store.save_json(
        pathlib.Path(work.config["ent"]["store_path"]),
    )

    work.ctx.save_erkg(
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )


    # finally, report the performance profiler stats
    if profiling:
        prof.analyze()
