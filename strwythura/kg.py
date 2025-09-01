#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Construct the lexical graph and condense it into a knowledge graph.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing

from icecream import ic  # type: ignore
import networkx as nx
import polars as pl
import spacy

from .context import DomainContext
from .elem import Entity, NodeKind, StrwVocab, TextChunk
from .nlp import Parser
from .scrape import Scraper
from .textrank import run_textrank, cooccur_entities


class KnowledgeGraph:
    """
Construct a _knowledge graph_ and build out assets to serialize then
use later.
    """

    def __init__ (
        self,
        config: dict,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config


    def build_graph (  # pylint: disable=R0912,R0913,R0914,R0917,W0102
        self,
        url_list: typing.List[ str ],
        domain_context: DomainContext,
        parser: Parser,
        simple_pipe: spacy.Language,
        entity_pipe: spacy.Language,
        *,
        debug: bool = False,
        ) -> None:
        """
Construct a knowledge graph from unstructured data sources.
        """
        label_map: typing.Dict[ str, str ] = domain_context.get_label_map()

        # iterate through the URL list, scraping text and building chunks
        scraper: Scraper = Scraper(self.config, parser)
        chunk_id: int = domain_context.start_chunk_id

        for url in url_list:
            # define data structures intialized for each parsed document
            lex_graph: nx.MultiDiGraph = nx.MultiDiGraph()
            chunk_list: typing.List[ TextChunk ] = []

            chunk_id = scraper.scrape_html(
                simple_pipe,
                url,
                chunk_list,
                chunk_id,
            )

            domain_context.chunk_table.add(chunk_list)  # type: ignore

            # parse each chunk to build a lexical graph per source URL
            for chunk in chunk_list:
                span_decoder: typing.Dict[ tuple, Entity ] = {}

                doc: spacy.tokens.doc.Doc = parser.parse_text(  # pylint: disable=I1101
                    domain_context,
                    entity_pipe,
                    lex_graph,
                    chunk,
                    debug = debug,
                )

                if debug:
                    ic(chunk)

                # keep track of sentence numbers per chunk, to use later
                # for entity co-occurrence links
                sent_map: typing.Dict[ spacy.tokens.span.Span, int ] = {}  # pylint: disable=I1101

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
                    self.make_entity(
                        domain_context,
                        span_decoder,
                        sent_map,
                        span,
                        label_map[span.label_], # decoded as abbrev IRI
                        chunk,
                        debug = debug,
                    )

                for span in doc.noun_chunks:
                    self.make_entity(
                        domain_context,
                        span_decoder,
                        sent_map,
                        span,
                        "NP",
                        chunk,
                        debug = False, # debug
                    )

                # overlay the recognized entity spans atop the base layer
                # constructed by _textgraph_ analysis of the `spaCy` parse trees
                for ent in span_decoder.values():
                    if ent.key not in parser.STOP_WORDS:
                        self.extract_entity(
                            domain_context,
                            lex_graph,
                            ent,
                            debug = debug,
                        )

                # extract relations for co-occurring entity pairs
                ## PLACEHOLDER

                # connect entities which co-occur within the same sentence
                cooccur_entities(
                    lex_graph,
                    span_decoder,
                )

                domain_context.add_entity_sequence(span_decoder)

            # apply _textrank_ to the graph (in the url/doc iteration)
            # then report the top-ranked extracted entities
            df: pl.DataFrame = run_textrank(
                self.config,
                lex_graph,
            )

            if debug:
                ic(url, df.head(11))

            # abstract a semantic overlay from the lexical graph
            # and persist this in the resulting KG
            self.abstract_overlay(
                domain_context,
                url,
                chunk_list,
                lex_graph,
            )

            if debug:
                print(
                    "nodes",
                    len(domain_context.sem_layer.nodes),
                    "edges",
                    len(domain_context.sem_layer.edges),
                )


    def abstract_overlay (  # pylint: disable=R0912,R0914
        self,
        domain_context: DomainContext,
        url: str,
        chunk_list: typing.List[ TextChunk ],
        lex_graph: nx.MultiDiGraph,
        ) -> None:
        """
Abstract a _semantic overlay_ from the lexical graph -- in other words
which nodes and edges get promoted up to the next level?

Also connect the extracted entities with their source chunks, where
the latter first-class citizens within the KG.
        """
        kept_nodes: typing.Set[ int ] = set()

        skipped_rel: typing.Set[ str ] = set([
            StrwVocab.CO_OCCURS_WITH.value,
            StrwVocab.COMPOUND_ELEM_OF.value,
            StrwVocab.FOLLOWS_LEXICALLY.value,
        ])

        chunk_nodes: typing.Dict[ int, str ] = {
            chunk.uid: f"chunk_{chunk.uid}"
            for chunk in chunk_list
        }

        for chunk_id, node_id in chunk_nodes.items():
            domain_context.sem_layer.add_node(
                node_id,
                kind = NodeKind.CHUNK.value,
                chunk = chunk_id,
                url = url,
            )

        for node_id, node_attr in lex_graph.nodes(data = True):
            if node_attr["kind"] == NodeKind.ENTITY.value:
                kept_nodes.add(node_id)
                count: int = node_attr["count"]

                if not domain_context.sem_layer.has_node(node_id):
                    domain_context.sem_layer.add_node(
                        node_id,
                        kind = NodeKind.ENTITY.value,
                        key = node_attr["key"],
                        text = node_attr["text"],
                        label = node_attr["label"],
                        rank = round(node_attr["rank"], 4),
                        count = count,
                    )
                else:
                    domain_context.sem_layer.nodes[node_id]["count"] += count

                domain_context.sem_layer.add_edge(
                    node_id,
                    chunk_nodes[node_attr["chunk"]],
                    key = StrwVocab.WITHIN_CHUNK.value,
                    weight = round(node_attr["rank"], 4),
                )

                # link each entity to its taxonomy concept,
                # though be careful not to introduce cycles
                if node_attr["label"] not in [ "NP" ]:
                    taxo_node_id: int = domain_context.taxo_node[node_attr["label"]]

                    if node_id != taxo_node_id:
                        domain_context.sem_layer.add_edge(
                            node_id,
                            taxo_node_id,
                            key = "RDF:type",
                            weight = 0.0
                        )

        for src_id, dst_id, key, edge_attr in lex_graph.edges(data = True, keys = True):
            if src_id in kept_nodes and dst_id in kept_nodes:
                prob: float = 1.0

                if "prob" in edge_attr:
                    prob = edge_attr["prob"]

                if key not in skipped_rel:
                    if not domain_context.sem_layer.has_edge(src_id, dst_id):
                        domain_context.sem_layer.add_edge(
                            src_id,
                            dst_id,
                            key = key,
                            prob = prob,
                        )
                    else:
                        domain_context.sem_layer.edges[src_id, dst_id, key]["prob"] = max(
                            prob,
                            domain_context.sem_layer.edges[src_id, dst_id, key]["prob"],
                        )


    def make_entity (  # pylint: disable=R0913,R0917
        self,
        domain_context: DomainContext,
        span_decoder: typing.Dict[ tuple, Entity ],
        sent_map: typing.Dict[ spacy.tokens.span.Span, int ],  # pylint: disable=I1101
        span: spacy.tokens.span.Span,  # pylint: disable=I1101
        label: str,
        chunk: TextChunk,
        *,
        debug: bool = False,  # pylint: disable=W0613
        ) -> Entity:
        """
Instantiate one `Entity` object, adding to our working "vocabulary".
        """
        lemma_key: str = domain_context.parse_lemma(span)  # type: ignore

        ent: Entity = Entity(
            ( span.start, span.end, ),
            lemma_key,
            span.text,
            label,
            chunk.uid,
            sent_map[span.sent],
            span,
        )

        if ent.loc not in span_decoder:
            span_decoder[ent.loc] = ent

            if False: # debug  # pylint: disable=W0125
                ic(ent)

        return ent


    def extract_entity (
        self,
        domain_context: DomainContext,
        lex_graph: nx.MultiDiGraph,
        ent: Entity,
        *,
        debug: bool = False,  # pylint: disable=W0613
        ) -> None:
        """
Link one `Entity` into this doc's lexical graph.
        """
        prev_known: bool = domain_context.add_lemma(ent.key)
        node_id: int = domain_context.get_lemma_index(ent.key)
        ent.node = node_id

        # hydrate a compound phrase in this doc's lexical graph
        if not lex_graph.has_node(node_id):
            lex_graph.add_node(
                node_id,
                key = ent.key,
                kind = NodeKind.ENTITY.value,
                label = ent.label,
                pos = "NP",
                text = ent.text,
                chunk = ent.chunk_id,
                count = 1,
            )

            for tok in ent.span:
                tok_lemma_key: str = domain_context.parse_lemma([ tok ])  # type: ignore

                if tok_lemma_key in domain_context.known_lemma:
                    tok_idx: int = domain_context.get_lemma_index(tok_lemma_key)

                    lex_graph.add_edge(
                        node_id,
                        tok_idx,
                        key = StrwVocab.COMPOUND_ELEM_OF.value,
                    )

        if prev_known:
            # promote a previous Lemma node to an Entity
            node: dict = lex_graph.nodes[node_id]
            node["kind"] = NodeKind.ENTITY.value
            node["chunk"] = ent.chunk_id
            node["count"] += 1

            # select the more specific label
            if "label" not in node or node["label"] == "NP":
                node["label"] = ent.label

        if False: # debug  # pylint: disable=W0125
            ic(ent)
