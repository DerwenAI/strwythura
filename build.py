#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphGeeks.org talk 2024-08-14 https://live.zoho.com/PBOB6fvr6c
How to construct _knowledge graphs_ from unstructured data sources.

This `build.py` script builds assets for constructing a KG then running
GraphRAG downstream.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing
import tracemalloc

from strwythura import Strwythura, PerfProfiler


if __name__ == "__main__":
    # start the performance profiling
    profiler: PerfProfiler = PerfProfiler()
    profiler.start()

    # construct the KG and build the assets for running GraphRAG later
    url_list: typing.List[ str ] = [
        "https://aaic.alz.org/releases-2024/processed-red-meat-raises-risk-of-dementia.asp",
        "https://www.theguardian.com/society/article/2024/jul/31/eating-processed-red-meat-could-increase-risk-of-dementia-study-finds",
        "https://www.massgeneralbrigham.org/en/about/newsroom/press-releases/red-meat-increases-risk-of-dementia",
        "https://www.alz.org/alzheimers-dementia/what-is-dementia",
    ]

    strw: Strwythura = Strwythura()

    strw.build_assets(
        url_list,
        debug = True,
    )

    # generate an interactive visualization in HTML
    strw.gen_visualization()

    # report the performance profiler stats
    profiler.report()
