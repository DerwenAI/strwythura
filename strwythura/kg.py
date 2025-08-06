#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Construct the knowledge graph.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import defaultdict
import typing

from icecream import ic
import lancedb
import networkx as nx
import pandas as pd
import spacy

from .lex import  STOP_WORDS, parse_text, extract_entity, extract_relations, make_entity
from .nlp import init_nlp_pipe
from .scrape import scrape_html
from .textrank import run_textrank, cooccur_entities
from .valid import Entity, TextChunk


def abstract_overlay (
    url: str,
    chunk_list: typing.List[ TextChunk ],
    lex_graph: nx.Graph,
    sem_overlay: nx.Graph,
    ) -> None:
    """
Abstract a _semantic overlay_ from the lexical graph -- in other words
which nodes and edges get promoted up to the next level?

Also connect the extracted entities with their source chunks, where
the latter first-class citizens within the KG.
    """
    kept_nodes: typing.Set[ int ] = set()
    skipped_rel: typing.Set[ str ] = set([ "FOLLOWS_LEXICALLY", "COMPOUND_ELEMENT_OF" ])

    chunk_nodes: typing.Dict[ int, str ] = {
        chunk.uid: f"chunk_{chunk.uid}"
        for chunk in chunk_list
    }

    for chunk_id, node_id in chunk_nodes.items():
        sem_overlay.add_node(
            node_id,
            kind = "Chunk",
            chunk = chunk_id,
            url = url,
        )

    for node_id, node_attr in lex_graph.nodes(data = True):
        if node_attr["kind"] == "Entity":
            kept_nodes.add(node_id)                
            count: int = node_attr["count"]

            if not sem_overlay.has_node(node_id):
                sem_overlay.add_node(
                    node_id,
                    kind = "Entity",
                    key = node_attr["key"],
                    text = node_attr["text"],
                    label = node_attr["label"],
                    count = count,
                )
            else:
                sem_overlay.nodes[node_id]["count"] += count

            sem_overlay.add_edge(
                node_id,
                chunk_nodes[node_attr["chunk"]],
                rel = "WITHIN",
                weight = node_attr["rank"],
            )

    for src_id, dst_id, edge_attr in lex_graph.edges(data = True):
        if src_id in kept_nodes and dst_id in kept_nodes:
            rel: str = edge_attr["rel"]
            prob: float = 1.0

            if "prob" in edge_attr:
                prob = edge_attr["prob"]

            if rel not in skipped_rel:
                if not sem_overlay.has_edge(src_id, dst_id):
                    sem_overlay.add_edge(
                        src_id,
                        dst_id,
                        rel = rel,
                        prob = prob,
                    )
                else:
                    sem_overlay[src_id][dst_id]["prob"] = max(
                        prob,
                        sem_overlay.edges[(src_id, dst_id)]["prob"],
                    )


def construct_kg (
    url_list: typing.List[ str ],
    simple_pipe: spacy.Language,
    chunk_table: lancedb.table.LanceTable,
    sem_overlay: nx.Graph,
    w2v_vectors: list = [],
    *,
    debug: bool = False,
    ) -> None:
    """
Construct a knowledge graph from unstructured data sources.
    """
    # define the global data structures which must be reset for each
    # run, not on each chunk iteration
    nlp_pipe: spacy.Language = init_nlp_pipe()
    known_lemma: typing.List[ str ] = []

    # iterate through the URL list, scraping text and building chunks
    chunk_id: int = 0

    for url in url_list:
        lex_graph: nx.Graph = nx.Graph()
        chunk_list: typing.List[ TextChunk ] = []

        chunk_id = scrape_html(
            simple_pipe,
            url,
            chunk_list,
            chunk_id,
        )

        chunk_table.add(chunk_list)

        # parse each chunk to build a lexical graph per source URL
        for chunk in chunk_list:
            span_decoder: typing.Dict[ tuple, Entity ] = {}

            doc: spacy.tokens.doc.Doc = parse_text(
                nlp_pipe,
                known_lemma,
                lex_graph,
                chunk,
                debug = debug,
            )

            if debug:
                ic(chunk)

            # keep track of sentence numbers per chunk, to use later
            # for entity co-occurrence links
            sent_map: typing.Dict[ spacy.tokens.span.Span, int ] = {}

            for sent_id, sent in enumerate(doc.sents):
                sent_map[sent] = sent_id

            # classify the recognized spans within this chunk as
            # potential entities

            # NB: if we'd run [_entity resolution_]
            # see: <https://neo4j.com/developer-blog/entity-resolved-knowledge-graphs/>
            # previously from _structured_ or _semi-structured_ data sources to
            # generate a "backbone" for the knowledge graph, then we could use
            # contextualized _surface forms_ perform _entity linking_ on the
            # entities extracted here from _unstructured_ data

            for span in doc.ents:
                make_entity(
                    span_decoder,
                    sent_map,
                    span,
                    chunk,
                    debug = debug,
                )

            for span in doc.noun_chunks:
                make_entity(
                    span_decoder,
                    sent_map,
                    span,
                    chunk,
                    debug = False, # debug
                )

            # overlay the recognized entity spans atop the base layer
            # constructed by _textgraph_ analysis of the `spaCy` parse trees
            for ent in span_decoder.values():
                if ent.key not in STOP_WORDS:
                    extract_entity(
                        known_lemma,
                        lex_graph,
                        ent,
                        debug = debug,
                    )

            # extract relations for co-occurring entity pairs
            extract_relations(
                known_lemma,
                lex_graph,
                span_decoder,
                sent_map,
                doc,
                chunk,
                debug = debug,
            )

            # connect entities which co-occur within the same sentence
            cooccur_entities(
                lex_graph,
                span_decoder,
            )

            # build the vector input for entity embeddings
            w2v_map: typing.Dict[ int, typing.Set[ str ]] = defaultdict(set)

            for ent in span_decoder.values():
                if ent.node is not None:
                    w2v_map[ent.sent_id].add(ent.key)

            for sent_id, ents in w2v_map.items():
                vec: list = list(ents)
                vec.insert(0, str(sent_id))
                w2v_vectors.append(vec)

        # apply _textrank_ to the graph (in the url/doc iteration)
        # then report the top-ranked extracted entities
        df: pd.DataFrame = run_textrank(
            lex_graph,
        )

        if debug:
            ic(url, df.head(20))

        # abstract a semantic overlay from the lexical graph
        # and persist this in the resulting KG
        abstract_overlay(
            url,
            chunk_list,
            lex_graph,
            sem_overlay,
        )

        if debug:
            print("nodes", len(sem_overlay.nodes), "edges", len(sem_overlay.edges))
