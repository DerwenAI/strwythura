#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Part 1: use the `sz_semantics` library for gRPC client/server access
to the Senzing SDK, to run entity resolution.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib

from strwythura import Profiler, Workflow
from sz_semantics import SzClient


if __name__ == "__main__":
    # start the performance profiling -
    profiling: bool = True # False

    if profiling:
        prof: Profiler = Profiler()


    # instantiate and configure our workflow manager
    work: Workflow = Workflow(
        config_path = pathlib.Path("config.toml"),
    )

    domain: dict = json.load(
        pathlib.Path("domain.json").open("r", encoding = "utf-8")
    )

    # access the Senzing SDK (running with a free-tier license)
    # and configure the namespaces for datasets
    sz: SzClient = SzClient(
        work.config,
        domain["sources"]["data"],
        debug = False,
    )

    # run entity resolution on the collection of datasets
    ents_batch: dict[ str, str ] = sz.entity_resolution(
        domain["sources"]["data"],
        debug = False,
    )

    print(json.dumps(ents_batch, indent = 2))

    # serialize the "GET_ENTITY" results for all resolved entities as
    # a JSONL file
    er_path: pathlib.Path = pathlib.Path(work.config["sz"]["er_path"])

    with er_path.open("w", encoding = "utf-8") as fp:
        for ent_json in sz.sz_engine.export_json_entity_report_iterator():
            fp.write(ent_json)

    # Production use cases would typically integrate the Senzing SDK
    # directly, probably in a data engineering workflow manager, via
    # pub-sub, etc.

    # However in this tutorial we're using the `export.json` file as
    # an intermediate -- so we can examine the results.


    # finally, report the performance profiler stats
    if profiling:
        prof.analyze()
