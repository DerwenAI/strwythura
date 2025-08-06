#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphGeeks.org talk 2024-08-14 https://live.zoho.com/PBOB6fvr6c
How to construct _knowledge graphs_ from unstructured data sources.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import defaultdict
import enum
import itertools
import json
import math
import pathlib
import sys
import traceback
import tracemalloc
import typing
import warnings

from bs4 import BeautifulSoup
from icecream import ic
from pyinstrument import Profiler
import gensim
import lancedb
import networkx as nx
import numpy as np
import pandas as pd
import requests
import spacy

from strwythura import Entity, TextChunk, GraphRAG, \
    KG_PATH, LANCEDB_URI, SPACY_MODEL, W2V_PATH, \
    RE_LABELS, init_nlp_pipe, make_chunk, \
    gen_pyvis, \
    calc_quantile_bins, stripe_column, root_mean_square


######################################################################
## define the model selections and parameter settings

SCRAPE_HEADERS: typing.Dict[ str, str ] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
}

STOP_WORDS: typing.Set[ str ] = set([
    "PRON.it",
    "PRON.that",
    "PRON.they",
    "PRON.those",
    "PRON.we",
    "PRON.which",
    "PRON.who",
])

TR_ALPHA: float = 0.85
TR_LOOKBACK: int = 3


######################################################################
## collect unstructured data from specific web page sources

def scrape_html (
    simple_pipe: spacy.Language,
    url: str,
    chunk_list: typing.List[ TextChunk ],
    chunk_id: int,
    ) -> int:
    """
A simple web page text scraper, which also performs chunking.
Returns the updated `chunk_id` index.
    """
    response: requests.Response = requests.get(
        url,
        headers = SCRAPE_HEADERS,
    )

    soup: BeautifulSoup = BeautifulSoup(
        response.text,
        features = "lxml",
    )

    scrape_doc: spacy.tokens.doc.Doc = simple_pipe("\n".join([
        para.text.strip()
        for para in soup.find_all("p")
    ]))

    chunk_id = make_chunk(
        scrape_doc,
        url,
        chunk_list,
        chunk_id,
    )

    return chunk_id


######################################################################
## lexical graph construction

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


######################################################################
## textrank algorithm for co-occurence and node ranking

def connect_entities (
    lex_graph: nx.Graph,
    span_decoder: typing.Dict[ tuple, Entity ],
    ) -> None:
    """
Connect entities which co-occur within the same sentence.
    """
    ent_map: typing.Dict[ int, typing.Set[ int ]] = defaultdict(set)

    for ent in span_decoder.values():
        if ent.node is not None:
            ent_map[ent.sent_id].add(ent.node)    

    for sent_id, nodes in ent_map.items():
        for pair in itertools.combinations(list(nodes), 2):
            if not lex_graph.has_edge(*pair):
                lex_graph.add_edge(
                    pair[0],
                    pair[1],
                    rel = "CO_OCCURS_WITH",
                    prob = 1.0,
                )


def run_textrank (
    lex_graph: nx.Graph,
    ) -> pd.DataFrame:
    """
Run eigenvalue centrality (i.e., _Personalized PageRank_) to rank the entities.
    """
    # build a dataframe of node ranks and counts
    df_rank: pd.DataFrame = pd.DataFrame.from_dict([
        {
            "node_id": node,
            "weight": rank,
            "count": lex_graph.nodes[node]["count"],
        }
        for node, rank in nx.pagerank(lex_graph, alpha = TR_ALPHA, weight = "count").items()
    ])

    # normalize by column and calculate quantiles
    df1: pd.DataFrame = df_rank[[ "count", "weight" ]].apply(lambda x: x / x.max(), axis = 0)
    bins: np.ndarray = calc_quantile_bins(len(df1.index))

    # stripe each columns
    df2: pd.DataFrame = pd.DataFrame([
        stripe_column(values, bins)
        for _, values in df1.items()
    ]).T

    # renormalize the ranks
    df_rank["rank"] = df2.apply(root_mean_square, axis=1)
    rank_col: np.ndarray = df_rank["rank"].to_numpy()
    rank_col /= sum(rank_col)
    df_rank["rank"] = rank_col

    # move the ranked weights back into the graph
    for _, row in df_rank.iterrows():
        node: int = row["node_id"]
        lex_graph.nodes[node]["rank"] = row["rank"]

    df: pd.DataFrame = pd.DataFrame([
        node_attr
        for node, node_attr in lex_graph.nodes(data = True)
        if node_attr["kind"] == "Entity"
    ]).sort_values(by = [ "rank" ], ascending = False)

    return df


######################################################################
## abstracting the semantic overlay out of the lexical graph

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
            connect_entities(
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


def train_entity_model (
    w2v_vectors: list,
    w2v_file: pathlib.Path,
    *,
    debug: bool = False,
    ) -> gensim.models.Word2Vec:
    """
Train a `gensim.Word2Vec` model for entity embeddings.
    """
    w2v_max: int = max([
        len(vec) - 1
        for vec in w2v_vectors
    ])

    w2v_model: gensim.models.Word2Vec = gensim.models.Word2Vec(
        w2v_vectors,
        min_count = 2,
        window = w2v_max,
    )

    w2v_model.save(str(w2v_file))

    return w2v_model


def main (
    debug: bool = False,
    ) -> int:
    """
Main entry point.
    """
    # start the stochastic call trace profiler and memory profiler
    profiler: Profiler = Profiler()
    profiler.start()
    tracemalloc.start()

    # define the global data structures
    url_list: typing.List[ str ] = [
        "https://aaic.alz.org/releases-2024/processed-red-meat-raises-risk-of-dementia.asp",
        "https://www.theguardian.com/society/article/2024/jul/31/eating-processed-red-meat-could-increase-risk-of-dementia-study-finds",
    ]

    simple_pipe: spacy.Language = spacy.load(SPACY_MODEL)

    vect_db: lancedb.db.LanceDBConnection = lancedb.connect(LANCEDB_URI)

    chunk_table: lancedb.table.LanceTable = vect_db.create_table(
        "chunk",
        schema = TextChunk,
        mode = "overwrite",
    )

    sem_overlay: nx.Graph = nx.Graph()

    try:
        w2v_vectors: list = []

        construct_kg(
            url_list,
            simple_pipe,
            chunk_table,
            sem_overlay,
            w2v_vectors,
            debug = debug,
        )

        # serialize the resulting KG
        with pathlib.Path(KG_PATH).open("w", encoding = "utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(sem_overlay, edges = "links"),
                    indent = 2,
                    sort_keys = True,
                )
            )

        # generate HTML for an interactive visualization of a graph
        gen_pyvis(
            sem_overlay,
            "kg.html",
            num_docs = len(url_list),
        )

        # train an entity embedding model
        w2v_model: gensim.models.Word2Vec = train_entity_model(
            w2v_vectors,
            W2V_PATH,
            debug = debug,
        )

        # run example queries
        rag: GraphRAG = GraphRAG(
            chunk_table,
            w2v_model,
            sem_overlay,
            )

        queries: typing.List[ str ] = [
            "dementia",
            "cognitive decline",
        ]

        for query in queries:
            entity: str = " ".join([
                f"{token.pos_}.{token.lemma_}"
                for token in simple_pipe(query)
            ])

            rag.get_chunks(
                query,
                [ entity ],
                debug = debug,
            )
    except Exception as ex:
        ic(ex)
        traceback.print_exc()

    # stop the profiler and report performance statistics
    profiler.stop()
    profiler.print()

    # report the memory usage
    report: tuple = tracemalloc.get_traced_memory()
    peak: float = round(report[1] / 1024.0 / 1024.0, 2)
    print(f"peak memory usage: {peak} MB")


######################################################################
## main entry point

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(debug = True)
