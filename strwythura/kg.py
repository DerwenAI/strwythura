#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Construct the knowledge graph.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import defaultdict
import typing

from icecream import ic  # type: ignore
import lancedb  # type: ignore
import networkx as nx
import polars as pl
import spacy

from .graph import Entity, TextChunk
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
        self.known_lemma: typing.List[ str ] = []


    def build_graph (  # pylint: disable=R0912,R0913,R0914,R0917,W0102
        self,
        parser: Parser,
        simple_pipe: spacy.Language,
        entity_pipe: spacy.Language,
        chunk_table: lancedb.table.LanceTable,
        sem_overlay: nx.Graph,
        w2v_vectors: list = [],
        *,
        debug: bool = False,
        ) -> None:
        """
Construct a knowledge graph from unstructured data sources.
        """
        # iterate through the URL list, scraping text and building chunks
        scraper: Scraper = Scraper(self.config, parser)
        chunk_id: int = 0

        for url in parser.url_list:
            # define data structures intialized for each parsed document
            lex_graph: nx.Graph = nx.Graph()
            chunk_list: typing.List[ TextChunk ] = []

            chunk_id = scraper.scrape_html(
                simple_pipe,
                url,
                chunk_list,
                chunk_id,
            )

            chunk_table.add(chunk_list)

            # parse each chunk to build a lexical graph per source URL
            for chunk in chunk_list:
                span_decoder: typing.Dict[ tuple, Entity ] = {}

                doc: spacy.tokens.doc.Doc = parser.parse_text(  # pylint: disable=I1101
                    entity_pipe,
                    self.known_lemma,
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
                        span_decoder,
                        sent_map,
                        span,
                        chunk,
                        debug = debug,
                    )

                for span in doc.noun_chunks:
                    self.make_entity(
                        span_decoder,
                        sent_map,
                        span,
                        chunk,
                        debug = False, # debug
                    )

                # overlay the recognized entity spans atop the base layer
                # constructed by _textgraph_ analysis of the `spaCy` parse trees
                for ent in span_decoder.values():
                    if ent.key not in parser.STOP_WORDS:
                        self.extract_entity(
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
            df: pl.DataFrame = run_textrank(
                self.config,
                lex_graph,
            )

            if debug:
                ic(url, df.head(11))

            # abstract a semantic overlay from the lexical graph
            # and persist this in the resulting KG
            self.abstract_overlay(
                url,
                chunk_list,
                lex_graph,
                sem_overlay,
            )

            if debug:
                print("nodes", len(sem_overlay.nodes), "edges", len(sem_overlay.edges))


    def abstract_overlay (  # pylint: disable=R0914
        self,
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
        skipped_rel: typing.Set[ str ] = set([
            "FOLLOWS_LEXICALLY",
            "COMPOUND_ELEMENT_OF",
        ])

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
                        rank = node_attr["rank"],
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


    def make_entity (  # pylint: disable=R0913
        self,
        span_decoder: typing.Dict[ tuple, Entity ],
        sent_map: typing.Dict[ spacy.tokens.span.Span, int ],  # pylint: disable=I1101
        span: spacy.tokens.span.Span,  # pylint: disable=I1101
        chunk: TextChunk,
        *,
        debug: bool = False,  # pylint: disable=W0613
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

            if False: # debug  # pylint: disable=W0125
                ic(ent)

        return ent


    def extract_entity (
        self,
        lex_graph: nx.Graph,
        ent: Entity,
        *,
        debug: bool = False,  # pylint: disable=W0613
        ) -> None:
        """
Link one `Entity` into this doc's lexical graph.
        """
        prev_known: bool = False

        if ent.key not in self.known_lemma:
            # add a new Entity node to the graph and link to its component Lemma nodes
            self.known_lemma.append(ent.key)
        else:
            # phrase for this entity has been previously seen in other documents
            prev_known = True

        node_id: int = self.known_lemma.index(ent.key)
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

                if tok_key in self.known_lemma:
                    tok_idx: int = self.known_lemma.index(tok_key)

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

        if False: # debug  # pylint: disable=W0125
            ic(ent)
