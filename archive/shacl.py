#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHACL tests
"""

import sys

from icecream import ic
import pyshacl
import rdflib


if __name__ == "__main__":
    debug: bool = False

    data_graph: str = "thesaurus.ttl"
    shacl_graph: str = "shacl.ttl"
    ont_graph: str | None = None

    conforms, results_graph, results_text = pyshacl.validate(
        data_graph,
        data_graph_format = "ttl",
        shacl_graph = shacl_graph,
        shacl_graph_format = "ttl",
        ont_graph = ont_graph,
        inference = 'rdfs',
        serialize_report_graph ="ttl",
        abort_on_first = False,
        allow_infos = False,
        allow_warnings = False,
        meta_shacl = False,
        advanced = False,
        js = False,
        debug = debug,
    )

    ic(conforms)
    ic(results_text)
    ic(results_graph)
