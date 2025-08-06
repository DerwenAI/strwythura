#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lexical graph construction.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import pathlib
import typing

from icecream import ic
import networkx as nx
import spacy

from .graph import Entity, TextChunk
from .nlp import RE_LABELS, STOP_WORDS
from .textrank import TR_LOOKBACK


def parse_text (
    nlp_pipe: spacy.Language,
    known_lemma: typing.List[ str ],
    lex_graph: nx.Graph,
    chunk: TextChunk,
    *,
    debug: bool = False,
    ) -> spacy.tokens.doc.Doc:
    """
Parse an input text chunk, returning a `spaCy` document.
    """
    doc: spacy.tokens.doc.Doc = list(
        nlp_pipe.pipe(
            [( chunk.text, RE_LABELS )],
            as_tuples = True,
        )
    )[0][0]

    # scan the document tokens to add lemmas to _lexical graph_ using
    # a _textgraph_ approach called the _textrank_ algorithm
    for sent in doc.sents:
        node_seq: typing.List[ int ] = []

        if False: # debug
            ic(sent)

        for tok in sent:
            text: str = tok.text.strip()
        
            if tok.pos_ in [ "NOUN", "PROPN" ]:
                key: str = tok.pos_ + "." + tok.lemma_.strip().lower()
                prev_known: bool = False
    
                if key not in known_lemma:
                    # create a new node
                    known_lemma.append(key)
                else:
                    # link to an existing node, adding weight
                    prev_known = True

                node_id: int = known_lemma.index(key)
                node_seq.append(node_id)

                if not lex_graph.has_node(node_id):
                    lex_graph.add_node(
                        node_id,
                        key = key,
                        kind = "Lemma",
                        pos = tok.pos_,
                        text = text,
                        chunk = chunk,
                        count = 1,
                    )

                elif prev_known:
                    node: dict = lex_graph.nodes[node_id]
                    node["count"] += 1

        # create the _textrank_ edges for the lexical graph,
        # which will get used for ranking, but discarded later
        if False: # debug
            ic(node_seq)

        for hop in range(TR_LOOKBACK):
            for node_id, node in enumerate(node_seq[: -1 - hop]):            
                neighbor: int = node_seq[hop + node_id + 1]
    
                if not lex_graph.has_edge(node, neighbor):
                    lex_graph.add_edge(
                        node,
                        neighbor,
                        rel = "FOLLOWS_LEXICALLY",
                    )

    return doc


def make_entity (
    span_decoder: typing.Dict[ tuple, Entity ],
    sent_map: typing.Dict[ spacy.tokens.span.Span, int ],
    span: spacy.tokens.span.Span,
    chunk: TextChunk,
    *,
    debug: bool = False,
    ) -> Entity:
    """
Instantiate one `Entity` object, adding to our working "vocabulary".
    """
    key: str = " ".join([
        tok.pos_ + "." + tok.lemma_.strip().lower()
        for tok in span
    ])

    ent: Entity = Entity(
        ( span.start, span.end, ),
        key,
        span.text,
        span.label_,
        chunk.uid,
        sent_map[span.sent],
        span,
    )

    if ent.loc not in span_decoder:
        span_decoder[ent.loc] = ent

        if False: # debug
            ic(ent)

    return ent


def extract_entity (
    known_lemma: typing.List[ str ],
    lex_graph: nx.Graph,
    ent: Entity,
    *,
    debug: bool = False,
    ) -> None:
    """
Link one `Entity` into this doc's lexical graph.
    """
    prev_known: bool = False

    if ent.key not in known_lemma:
        # add a new Entity node to the graph and link to its component Lemma nodes
        known_lemma.append(ent.key)
    else:
        # phrase for this entity has been previously seen in other documents
        prev_known = True

    node_id: int = known_lemma.index(ent.key)
    ent.node = node_id

    # hydrate a compound phrase in this doc's lexical graph
    if not lex_graph.has_node(node_id):
        lex_graph.add_node(
            node_id,
            key = ent.key,
            kind = "Entity",
            label = ent.label,
            pos = "NP",
            text = ent.text,
            chunk = ent.chunk_id,
            count = 1,
        )

        for tok in ent.span:
            tok_key: str = tok.pos_ + "." + tok.lemma_.strip().lower()

            if tok_key in known_lemma:
                tok_idx: int = known_lemma.index(tok_key)

                lex_graph.add_edge(
                    node_id,
                    tok_idx,
                    rel = "COMPOUND_ELEMENT_OF",
                )


    if prev_known:
        # promote a previous Lemma node to an Entity
        node: dict = lex_graph.nodes[node_id]
        node["kind"] = "Entity"
        node["chunk"] = ent.chunk_id
        node["count"] += 1

        # select the more specific label
        if "label" not in node or node["label"] == "NP":
            node["label"] = ent.label
    
    if False: # debug
        ic(ent)


def extract_relations (
    known_lemma: typing.List[ str ],
    lex_graph: nx.Graph,
    span_decoder: typing.Dict[ tuple, Entity ],
    sent_map: typing.Dict[ spacy.tokens.span.Span, int ],
    doc: spacy.tokens.doc.Doc,
    chunk: TextChunk,
    *,
    debug: bool = False,
    ) -> None:
    """
Extract the relations inferred by `GLiREL` adding these to the graph.
    """
    relations: typing.List[ dict ] = sorted(
        doc._.relations,
        key = lambda item: item["score"],
        reverse = True,
    )

    for item in relations:
        src_loc: typing.Tuple[ int ] = tuple(item["head_pos"])
        dst_loc: typing.Tuple[ int ] = tuple(item["tail_pos"])
        redact_rel: bool = False

        if src_loc not in span_decoder:
            if False: # debug
                print("MISSING src entity:", item["head_text"], item["head_pos"])

            src_ent: Entity = make_entity(
                span_decoder,
                sent_map,
                doc[ item["head_pos"][0] : item["head_pos"][1] ],
                chunk,
                debug = False, # debug
            )

            if src_ent.key in STOP_WORDS:
                redact_rel = True
            else:
                extract_entity(
                    known_lemma,
                    lex_graph,
                    src_ent,
                    debug = debug
                )

        if dst_loc not in span_decoder:
            if debug:
                print("MISSING dst entity:", item["tail_text"], item["tail_pos"])

            dst_ent: Entity = make_entity(
                span_decoder,
                sent_map,
                doc[ item["tail_pos"][0] : item["tail_pos"][1] ],
                chunk,
                debug = False, # debug
            )

            if dst_ent.key in STOP_WORDS:
                redact_rel = True
            else:
                extract_entity(
                    known_lemma,
                    lex_graph,
                    dst_ent,
                    debug = debug
                )

        # link the connected nodes
        if not redact_rel:
            src_ent = span_decoder[src_loc]
            dst_ent = span_decoder[dst_loc]

            rel: str = item["label"].strip().replace(" ", "_").upper()
            prob: float = round(item["score"], 3)

            if debug:
                print(f"{src_ent.text} -> {rel} -> {dst_ent.text} | {prob}")

            lex_graph.add_edge(
                src_ent.node,
                dst_ent.node,
                rel = rel,
                prob = prob,
            )
