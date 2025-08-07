#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphGeeks.org talk 2024-08-14 https://live.zoho.com/PBOB6fvr6c
How to construct _knowledge graphs_ from unstructured data sources.

This `demo.py` script builds assets for constructing a KG then running
GraphRAG downstream.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing
import tracemalloc

from pyinstrument import Profiler
from strwythura import Strwythura


if __name__ == "__main__":
    # start the profiling
    profiler: Profiler = Profiler()
    profiler.start()
    tracemalloc.start()

    # construct the KG and build the assets for GraphRAG later
    url_list: typing.List[ str ] = [
        "https://aaic.alz.org/releases-2024/processed-red-meat-raises-risk-of-dementia.asp",
        "https://www.theguardian.com/society/article/2024/jul/31/eating-processed-red-meat-could-increase-risk-of-dementia-study-finds",
        "https://www.massgeneralbrigham.org/en/about/newsroom/press-releases/red-meat-increases-risk-of-dementia",
    ]

    ner_labels: typing.List[ str ] = [
        "Behavior",
        "City",
        "Company",    
        "Condition",
        "Conference",
        "Country",
        "Food",
        "Food Additive",
        "Hospital",
        "Organ",
        "Organization",
        "People Group",
        "Person",
        "Publication",
        "Research",
        "Science",
        "University",
    ]

    strw: Strwythura = Strwythura()

    strw.build_assets(
        url_list,
        ner_labels,
        debug = True,
    )

    strw.gen_visualization()

    # stop the call trace profiler and report performance statistics
    profiler.stop()
    profiler.print()

    # report memory usage
    report: tuple = tracemalloc.get_traced_memory()
    peak: float = round(report[1] / 1024.0 / 1024.0, 2)
    print(f"peak memory usage: {peak} MB")
