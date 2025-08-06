#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphGeeks.org talk 2024-08-14 https://live.zoho.com/PBOB6fvr6c
How to construct _knowledge graphs_ from unstructured data sources.

This `demo.py` script builds assets for constructing a KG then running
GraphRAG downstream.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import tracemalloc
import warnings

from pyinstrument import Profiler
from strwythura import build_assets


if __name__ == "__main__":
    # start the profiling
    profiler: Profiler = Profiler()
    profiler.start()
    tracemalloc.start()

    # construct the KG and build the GraphRAG assets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_assets(debug = True)

    # stop the call trace profiler and report performance statistics
    profiler.stop()
    profiler.print()

    # report memory usage
    report: tuple = tracemalloc.get_traced_memory()
    peak: float = round(report[1] / 1024.0 / 1024.0, 2)
    print(f"peak memory usage: {peak} MB")
