#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphGeeks.org talk 2024-08-14 https://live.zoho.com/PBOB6fvr6c
How to construct _knowledge graphs_ from unstructured data sources.
"""

from collections import defaultdict
from dataclasses import dataclass
import itertools
import os
import sys
import typing
import unicodedata
import warnings

from bs4 import BeautifulSoup
from gliner_spacy.pipeline import GlinerSpacy
from icecream import ic
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import glirel
import lancedb
import networkx as nx
import pandas as pd
import pyvis
import requests
import spacy
import transformers


######################################################################
## define the model selections and parameter settings

CHUNK_SIZE: int = 1024

EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

EMBED_FCN: lancedb.embeddings.transformers.TransformersEmbeddingFunction = \
    get_registry().get("huggingface").create(name = EMBED_MODEL)

GLINER_MODEL: str = "urchade/gliner_small-v2.1"

LANCEDB_URI = "data/lancedb"

NER_LABELS: typing.List[ str] = [
    "Company",
    "Condition",
    "Food",
    "Location",
    "Organization",
    "Person",
    "Research",
]

RE_LABELS: dict = {
    "glirel_labels": {
        "co_founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]}, 
        "country_of_origin": {"allowed_head": ["PERSON", "ORG"], "allowed_tail": ["LOC", "GPE"]}, 
        "no_relation": {},  
        "parent": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]}, 
        "followed_by": {"allowed_head": ["PERSON", "ORG"], "allowed_tail": ["PERSON", "ORG"]},  
        "spouse": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},  
        "child": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},  
        "founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},  
        "headquartered_in": {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE", "FAC"]},  
        "acquired_by": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},  
        "subsidiary_of": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]}, 
    }
}

SPACY_MODEL: str = "en_core_web_md"

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
## data validation classes

class TextChunk (LanceModel):
    uid: int
    url: str
    text: str = EMBED_FCN.SourceField()
    vector: Vector(EMBED_FCN.ndims()) = EMBED_FCN.VectorField(default = None)


@dataclass(order=False, frozen=False)
class Entity:
    loc: typing.Tuple[ int ]
    key: str
    text: str
    label: str
    chunk_id: int
    sent_id: int
    span: spacy.tokens.span.Span
    node: typing.Optional[ int ] = None


######################################################################
## workflow function definitions

def init_nlp (
    ) -> spacy.Language:
    """
Initialize the models.
    """
    # override specific Hugging Face error messages, since
    # `transformers` and `tokenizers` have noisy logging
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "0"

    # load models for `spaCy`, `GLiNER`, `GLiREL`
    # this may take several minutes when run the first time
    nlp: spacy.Language = spacy.load(SPACY_MODEL)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        nlp.add_pipe(
            "gliner_spacy",
            config = {
                "gliner_model": GLINER_MODEL,
                "labels": NER_LABELS,
                "chunk_size": CHUNK_SIZE,
                "style": "ent",
            },
        )
        
        nlp.add_pipe(
            "glirel",
            after = "ner",
        )

    return nlp


def uni_scrubber (
    span: spacy.tokens.span.Span,
    ) -> str:
    """
Applies multiple approaches for aggressively removing garbled Unicode
and spurious punctuation from the given text.

OH: "It scrubs the garble from its stream... or it gets the debugger again!"
    """
    text: str = span.text

    if type(text).__name__ != "str":
        print("not a string?", type(text), text)

    limpio: str = " ".join(map(lambda s: s.strip(), text.split("\n"))).strip()

    limpio = limpio.replace('“', '"').replace('”', '"')
    limpio = limpio.replace("‘", "'").replace("’", "'").replace("`", "'").replace("â", "'")
    limpio = limpio.replace("…", "...").replace("–", "-")

    limpio = str(unicodedata.normalize("NFKD", limpio).encode("ascii", "ignore").decode("utf-8"))

    return limpio


def make_chunk (
    doc: spacy.tokens.doc.Doc,
    url: str,
    chunk_list: typing.List[ TextChunk ],
    chunk_id: int,
    ) -> int:
    """
Split the given document into text chunks, returning the last index.
BTW, for ideal text chunk size see
<https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5>
    """
    chunks: typing.List[ str ] = []
    chunk_total: int = 0
    prev_line: str = ""

    for sent_id, sent in enumerate(doc.sents):
        line: str = uni_scrubber(sent)
        line_len: int = len(line)
    
        if (chunk_total + line_len) > CHUNK_SIZE:
            # emit the current chunk
            chunk_list.append(
                TextChunk(
                    uid = chunk_id,
                    url = url,
                    text = "\n".join(chunks),
                )
            )

            # start a new chunk
            chunks = [ prev_line, line ]
            chunk_total = len(prev_line) + line_len
            chunk_id += 1
        else:
            # append line to the current chunk
            chunks.append(line)
            chunk_total += line_len

        prev_line = line

    # emit the trailing chunk
    chunk_list.append(
        TextChunk(
            uid = chunk_id,
            url = url,
            text = "\n".join(chunks),
        )
    )

    return chunk_id + 1


def parse_text (
    nlp: spacy.Language,
    known_lemma: typing.List[ str ],
    graph: nx.Graph,
    chunk: TextChunk,
    *,
    debug: bool = False,
    ) -> spacy.tokens.doc.Doc:
    """
Parse an input text chunk, returning a `spaCy` document.
    """
    doc: spacy.tokens.doc.Doc = list(
        nlp.pipe(
            [( chunk.text, RE_LABELS )],
            as_tuples = True,
        )
    )[0][0]

    # scan the document tokens to add lemmas to _lexical graph_ using
    # a _textgraph_ algorithm
    for sent in doc.sents:
        node_seq: typing.List[ int ] = []

        if debug:
            ic(sent)

        for tok in sent:
            text: str = tok.text.strip()
        
            if tok.pos_ in [ "NOUN", "PROPN" ]:
                key: str = tok.pos_ + "." + tok.lemma_.strip().lower()
    
                if key not in known_lemma:
                    # create a new node
                    known_lemma.append(key)
                    node_id: int = known_lemma.index(key)
                    node_seq.append(node_id)

                    graph.add_node(
                        node_id,
                        key = key,
                        kind = "Lemma",
                        pos = tok.pos_,
                        text = text,
                        count = 1,
                    )
                else:
                    # link to an existing node, adding weight
                    node_id = known_lemma.index(key)
                    node_seq.append(node_id)

                    node: dict = graph.nodes[node_id]
                    node["count"] += 1

        # create the textrank edges
        if debug:
            ic(node_seq)

        for hop in range(TR_LOOKBACK):
            for node_id, node in enumerate(node_seq[: -1 - hop]):            
                neighbor: int = node_seq[hop + node_id + 1]
    
                graph.add_edge(
                    node,
                    neighbor,
                    rel = "FOLLOWS_LEXICALLY",
                )

    return doc


def make_entity (
    span_decoder: typing.Dict[ tuple, Entity ],
    span: spacy.tokens.span.Span,
    chunk: TextChunk,
    *,
    debug: bool = False,
    ) -> Entity:
    """
Instantiate one `Entity` dataclass object, adding to our working "vocabulary".
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

        if debug:
            ic(ent)

    return ent


def extract_entity (
    known_lemma: typing.List[ str ],
    graph: nx.Graph,
    ent: Entity,
    *,
    debug: bool = False,
    ) -> None:
    """
Link one `Entity` into the existing graph.
    """
    if ent.key not in known_lemma:
        # add a new Entity node to the graph and link to its component Lemma nodes
        known_lemma.append(ent.key)
        node_id: int = known_lemma.index(ent.key)
        
        graph.add_node(
            node_id,
            key = ent.key,
            kind = "Entity",
            label = ent.label,
            pos = "NP",
            text = ent.text,
            count = 1,
        )

        for tok in ent.span:
            tok_key: str = tok.pos_ + "." + tok.lemma_.strip().lower()

            if tok_key in known_lemma:
                tok_idx: int = known_lemma.index(tok_key)

                graph.add_edge(
                    node_id,
                    tok_idx,
                    rel = "COMPOUND_ELEMENT_OF",
                )
    else:
        node_id: int = known_lemma.index(ent.key)
        node: dict = graph.nodes[node_id]
        # promote to an Entity, in case the node had been a Lemma
        node["kind"] = "Entity"
        node["label"] = ent.label
        node["count"] += 1
    
    ent.node = node_id

    if debug:
        ic(ent)


def extract_relations (
    known_lemma: typing.List[ str ],
    graph: nx.Graph,
    span_decoder: typing.Dict[ tuple, Entity ],
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
            if debug:
                print("MISSING src entity:", item["head_text"], item["head_pos"])

            src_ent: Entity = make_entity(
                span_decoder,
                doc[ item["head_pos"][0] : item["head_pos"][1] ],
                chunk,
                debug = debug,
            )

            if src_ent.key in STOP_WORDS:
                redact_rel = True
            else:
                extract_entity(
                    known_lemma,
                    graph,
                    src_ent,
                    debug = debug
                )

        if dst_loc not in span_decoder:
            if debug:
                print("MISSING dst entity:", item["tail_text"], item["tail_pos"])

            dst_ent: Entity = make_entity(
                span_decoder,
                doc[ item["tail_pos"][0] : item["tail_pos"][1] ],
                chunk,
                debug = debug,
            )

            if dst_ent.key in STOP_WORDS:
                redact_rel = True
            else:
                extract_entity(
                    known_lemma,
                    graph,
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

            graph.add_edge(
                src_ent.node,
                dst_ent.node,
                rel = rel,
                prob = prob,
            )


def connect_entities (
    graph: nx.Graph,
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
            graph.add_edge(
                pair[0],
                pair[1],
                rel = "CO_OCCURS_WITH",
                prob = 1.0,
            )


def run_textrank (
    graph: nx.Graph,
    ) -> pd.DataFrame:
    """
Run eigenvalue centrality (i.e., _Personalized PageRank_) to rank the entities.
    """
    for node, rank in nx.pagerank(graph, alpha = TR_ALPHA, weight = "count").items():
        graph.nodes[node]["rank"] = rank

    df: pd.DataFrame = pd.DataFrame([
        node_attr
        for node, node_attr in graph.nodes(data = True)
        if node_attr["kind"] == "Entity"
    ]).sort_values(by = [ "rank", "count" ], ascending = False)

    return df


def gen_pyvis (
    graph: nx.Graph,
    html_file: str,
    *,
    notebook: bool = False,
    ) -> None:
    """
Use `pyvis` to provide an interactive visualization of the graph layers.
    """
    pv_net: pyvis.network.Network = pyvis.network.Network(
        height = "750px",
        width = "100%",
        notebook = notebook,
        cdn_resources = "remote",
    )

    for node_id, node_attr in graph.nodes(data = True):
        if node_attr["kind"] == "Entity":
            color: str = "hsl(65, 46%, 58%)"
            size: int = round(200 * node_attr["rank"])
        else:
            color = "hsla(72, 10%, 90%, 0.95)"
            size = round(30 * node_attr["rank"])

        pv_net.add_node(
            node_id,
            label = node_attr["text"],
            title = node_attr.get("label"),
            color = color,
            size = size,
        )

    for src_node, dst_node, edge_attr in graph.edges(data = True):
        pv_net.add_edge(
            src_node,
            dst_node,
            title = edge_attr.get("rel"),
        )

        pv_net.toggle_physics(True)
        pv_net.show_buttons(filter_ = True)
        pv_net.save_graph(html_file)


if __name__ == "__main__":
    debug: bool = True # False

    # iterate through the URL list scraping text and building chunks
    vect_db: lancedb.db.LanceDBConnection = lancedb.connect(LANCEDB_URI)

    chunk_table: lancedb.table.LanceTable = vect_db.create_table(
        "chunk",
        schema = TextChunk,
        mode = "overwrite",
    )

    chunk_list: typing.List[ TextChunk ] = []
    chunk_id: int = 0

    scrape_nlp: spacy.Language = spacy.load(SPACY_MODEL)

    url_list: typing.List[ str ] = [
        "https://aaic.alz.org/releases-2024/processed-red-meat-raises-risk-of-dementia.asp",
        "https://www.theguardian.com/society/article/2024/jul/31/eating-processed-red-meat-could-increase-risk-of-dementia-study-finds",
    ]

    headers: typing.Dict[ str, str ] = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    }

    for url in url_list:
        response: requests.Response = requests.get(
            url,
            headers = headers,
        )

        soup: BeautifulSoup = BeautifulSoup(
            response.text,
            features = "lxml",
        )

        scrape_doc: spacy.tokens.doc.Doc = scrape_nlp("\n".join([
            para.text.strip()
            for para in soup.findAll("p")
        ]))

        chunk_id = make_chunk(
            scrape_doc,
            url,
            chunk_list,
            chunk_id,
        )

    chunk_table.add(chunk_list)

    # define the global data structures which must be reset for each
    # run, not on each chunk iteration
    nlp: spacy.Language = init_nlp()
    graph: nx.Graph = nx.Graph()
    known_lemma: typing.List[ str ] = []

    # iterate through the text chunks to parase and build the graph
    for chunk in chunk_list:
        doc: spacy.tokens.doc.Doc = parse_text(
            nlp,
            known_lemma,
            graph,
            chunk,
            debug = debug,
        )

        # keep track of sentence numbers, to use for entity co-occurrence
        # links
        sent_map: typing.Dict[ spacy.tokens.span.Span, int ] = {}

        for sent_id, sent in enumerate(doc.sents):
            sent_map[sent] = sent_id

        # classify the recognized spans within this chunk as potential
        # entities
        span_decoder: typing.Dict[ tuple, Entity ] = {}

        # NB: if we'd run [_entity resolution_](https://neo4j.com/developer-blog/entity-resolved-knowledge-graphs/)
        # previously from _structured_ or _semi-structured_ data sources to
        # generate a "backbone" for the knowledge graph, then we could use
        # contextualized _surface forms_ perform _entity linking_ on the
        # entities extracted here from _unstructured_ data

        for span in doc.ents:
            make_entity(
                span_decoder,
                span,
                chunk,
                debug = debug,
            )

        for span in doc.noun_chunks:
            make_entity(
                span_decoder,
                span,
                chunk,
                debug = debug,
            )

        # overlay the recognized entity spans atop the base layer
        # constructed by _textgraph_ analysis of the `spaCy` parse trees
        for ent in span_decoder.values():
            if ent.key not in STOP_WORDS:
                extract_entity(
                    known_lemma,
                    graph,
                    ent,
                    debug = debug,
                )

        # extract relations for co-occurring entity pairs
        extract_relations(
            known_lemma,
            graph,
            span_decoder,
            doc,
            chunk,
            debug = debug,
        )

        # connect entities which co-occur within the same sentence
        connect_entities(
            graph,
            span_decoder,
        )


    ######################################################################
    # apply _textrank_ then report the top-ranked extracted entities
    df: pd.DataFrame = run_textrank(
        graph,
    )

    ic(df.head(20))

    ######################################################################
    # generate HTML for an interactive visualization of the graph

    gen_pyvis(
        graph,
        "chunk.html",
    )
