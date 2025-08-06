#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Package definitions for Strwythura.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

KG_PATH: str = "data/kg.json"
LANCEDB_URI: str = "data/lancedb"
W2V_PATH: str = "data/entity.w2v"


from .nlp import NER_LABELS, RE_LABELS, SPACY_MODEL, \
    init_nlp_pipe, make_chunk, uni_scrubber

from .opt import calc_quantile_bins, stripe_column, root_mean_square

from .rag import GraphRAG

from .valid import Entity, TextChunk

from .vis import gen_pyvis

