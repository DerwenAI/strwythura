#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Package definitions for Strwythura.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

CHUNK_TABLE: str = "chunk"
HTML_PATH: str = "kg.html"
KG_PATH: str = "data/kg.json"
LANCEDB_URI: str = "data/lancedb"
W2V_PATH: str = "data/entity.w2v"


from .kg import abstract_overlay, construct_kg

from .lex import STOP_WORDS, \
    extract_entity, extract_relations, make_entity, \
    parse_text, train_entity_model

from .nlp import NER_LABELS, RE_LABELS, SPACY_MODEL, \
    init_nlp_pipe, make_chunk, uni_scrubber

from .opt import calc_quantile_bins, stripe_column, root_mean_square

from .rag import GraphRAG

from .scrape import scrape_html

from .textrank import TR_ALPHA, TR_LOOKBACK, \
    run_textrank, cooccur_entities

from .valid import Entity, TextChunk

from .vis import gen_pyvis

