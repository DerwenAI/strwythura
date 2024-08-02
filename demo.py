

from collections import defaultdict
from dataclasses import dataclass, field
import itertools
import os
import typing
import warnings

from gliner_spacy.pipeline import GlinerSpacy
from icecream import ic
import glirel
import networkx as nx
import pandas as pd
import pyvis
import spacy
import transformers


# override specific Hugging Face error messages, since `transformers`
# and `tokenizers` have noisy logging

transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "0"


# define the model selections and parameter settings

CHUNK_SIZE: int = 250

GLINER_MODEL: str = "urchade/gliner_small-v2.1"
GLINER_USED: bool = True # False

NER_LABELS: typing.List[ str] = [
    "Person",
    "Company",
    "Location",
    "Food",
    "Disease",
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
    "that",
    "they",
    "those",
])

TR_ALPHA: float = 0.85
TR_LOOKBACK: int = 3


# load the models for `spaCy`, `GLiNER`, `GLiREL` -- this may take several
# minutes when run the first time

nlp: spacy.Language = spacy.load(SPACY_MODEL)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    if GLINER_USED:
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


# define the global data structures -- which need to be reset for
# every run, not for each chunk iteration

graph: nx.Graph = nx.Graph()
known_lemma: typing.List[ str ] = []


# text chunking

@dataclass(order=False, frozen=False)
class TextChunk:
    uid: int
    url: str
    text: str

demen_chunk: TextChunk = TextChunk(
    2,
    "https://www.theguardian.com/society/article/2024/jul/31/eating-processed-red-meat-could-increase-risk-of-dementia-study-finds",
    """
Eating processed red meat could be a significant risk factor for dementia, according to a large study that tracked more than 100,000 people over four decades.
Processed red meat has previously been shown to increase the risk of cancer, heart disease and type 2 diabetes. Now US researchers say they have uncovered a potential link to dementia.
The study also found that replacing processed red meat with healthier foods such as nuts, beans or tofu could help reduce the risk of dementia. The findings were presented at the Alzheimer’s Association international conference in the US.
The number of people living with dementia globally is forecast to nearly triple to 153 million by 2050, and studies looking at diet and risk of cognitive decline has become a focus of researchers.

In the latest research, experts studied the health of 130,000 nurses and other health workers working in the US. They were tracked for 43 years and provided data on their diet every 2 to 5 years.
The participants were asked how often they ate processed red meat including bacon, hotdogs, sausages, salami and other sandwich meat.
They were also asked about their consumption of nuts and legumes including peanut butter, peanuts, walnuts and other nuts, string beans, beans, peas, soy milk and tofu.

More than 11,000 cases of dementia were identified during the follow-up period.
Consuming two servings of processed red meat each week appeared to raise the risk of cognitive decline by 14% compared with those eating about three servings a month, the researchers reported.
The study also suggested that replacing one daily serving of processed red meat for a daily serving of nuts, beans or tofu every day could lower the risk of dementia by 23%.
    """.strip(),
)

text_chunk: TextChunk = demen_chunk


# parse the input text

doc: spacy.tokens.doc.Doc = list(
    nlp.pipe(
        [( text_chunk.text, RE_LABELS )],
        as_tuples = True,
    )
)[0][0]


# scan the document tokens to add lemmas to _lexical graph_ using a
# _textgraph_ algorithm

for sent in doc.sents:
    node_seq: typing.List[ int ] = []
    ic(sent)

    for tok in sent:
        text: str = tok.text.strip()
        
        if tok.pos_ in [ "NOUN", "PROPN" ] and text not in STOP_WORDS:
            key: str = tok.pos_ + "." + tok.lemma_.strip().lower()
            print(tok.i, key, tok.text.strip())

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
                    chunk = text_chunk.uid,
                    count = 1,
                )
            else:
                # link to an existing node, adding weight
                node_id = known_lemma.index(key)
                node_seq.append(node_id)

                node: dict = graph.nodes[node_id]
                node["count"] += 1

    # create the textrank edges
    ic(node_seq)

    for hop in range(TR_LOOKBACK):
        for node_id, node in enumerate(node_seq[: -1 - hop]):            
            neighbor: int = node_seq[hop + node_id + 1]
            graph.add_edge(
                node,
                neighbor,
                rel = "FOLLOWS_LEXICALLY",
            )


# keep track of sentence numbers, to use later for entity co-occurrence links

sent_map: typing.Dict[ spacy.tokens.span.Span, int ] = {}

for sent_id, sent in enumerate(doc.sents):
    sent_map[sent] = sent_id


# classify spans as potential entities --
#
# NB: if we'd run [_entity resolution_](https://neo4j.com/developer-blog/entity-resolved-knowledge-graphs/)
# previously from _structured_ or _semi-structured_ data sources to
# generate a "backbone" for the knowledge graph, then we use the contextualized
# _surface forms_ from that phase to perform _entity linking_ on the
# entities extracted here from _unstructured_ data

@dataclass(order=False, frozen=False)
class Entity:
    loc: typing.Tuple[ int ]
    text: str
    label: str
    chunk_id: int
    sent_id: int
    span: spacy.tokens.span.Span
    node: typing.Optional[ int ] = None


span_decoder: typing.Dict[ tuple, Entity ] = {}


def make_entity (
    span: spacy.tokens.span.Span,
    chunk: TextChunk,
    ) -> Entity:
    """
Instantiate one `Entity` dataclass object, adding it to the working "vocabulary".
    """
    ent: Entity = Entity(
        ( span.start, span.end, ),
        span.text,
        span.label_,
        chunk.uid,
        sent_map[span.sent],
        span,
    )

    if ent.loc not in span_decoder:
        span_decoder[ent.loc] = ent
        ic(ent)

    return ent


for span in doc.ents:
    make_entity(span, text_chunk)

for span in doc.noun_chunks:
    make_entity(span, text_chunk)


# overlay the inferred entity spans atop the base layer constructed by
# _textgraph_ analysis of the `spaCy` parse trees

def add_entity (
    ent: Entity,
    ) -> None:
    """
Link one `Entity` into the existing graph.
    """
    key: str = " ".join([
        tok.pos_ + "." + tok.lemma_.strip().lower()
        for tok in ent.span
    ])

    if key not in known_lemma:
        # add a new Entity node to the graph and link to its component Lemma nodes
        known_lemma.append(key)
        node_id: int = known_lemma.index(key)
        
        graph.add_node(
            node_id,
            key = key,
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

                graph.add_edge(
                    node_id,
                    tok_idx,
                    rel = "COMPOUND_ELEMENT_OF",
                )
    else:
        node_id: int = known_lemma.index(key)
        node: dict = graph.nodes[node_id]
        # promote to an Entity, in case the node had been a Lemma
        node["kind"] = "Entity"
        node["label"] = ent.label
        node["count"] += 1
    
    ent.node = node_id


for ent in span_decoder.values():
    add_entity(ent)
    ic(ent)


# report the relations inferred by `GLiREL`

relations = sorted(
    doc._.relations,
    key = lambda x: x["score"],
    reverse = True,
)

for item in relations:
    src_loc: typing.Tuple[ int ] = tuple(item["head_pos"])
    dst_loc: typing.Tuple[ int ] = tuple(item["tail_pos"])

    if src_loc not in span_decoder:
        print("MISSING src entity:", item["head_text"], item["head_pos"])
        span: spacy.tokens.span.Span = doc[ item["head_pos"][0] : item["head_pos"][1] ]
        add_entity(make_entity(span, text_chunk))

    if dst_loc not in span_decoder:
        print("MISSING dst entity:", item["tail_text"], item["tail_pos"])
        span: spacy.tokens.span.Span = doc[ item["tail_pos"][0] : item["tail_pos"][1] ]
        add_entity(make_entity(span, text_chunk))

    # link the connected nodes
    src_ent = span_decoder[src_loc]
    dst_ent = span_decoder[dst_loc]

    rel: str = item["label"].strip().replace(" ", "_").upper()
    prob: float = round(item["score"], 3)

    print(f"{src_ent.text} -> {rel} -> {dst_ent.text} | {prob}")

    graph.add_edge(
        src_ent.node,
        dst_ent.node,
        rel = rel,
        prob = prob,
    )


# connect entities which co-occur within the same sentence

ent_map: typing.Dict[ int, typing.Set[ int ]] = defaultdict(set)

for ent in span_decoder.values():
    ent_map[ent.sent_id].add(ent.node)    

for sent_id, nodes in ent_map.items():
    for pair in itertools.combinations(list(nodes), 2):
        graph.add_edge(
            pair[0],
            pair[1],
            rel = "CO_OCCURS_WITH",
            prob = 1.0,
        )

# run eigenvalue centrality (i.e., _Personalized PageRank_) to rank the entities

for node, rank in nx.pagerank(graph, alpha = TR_ALPHA, weight = "count").items():
    graph.nodes[node]["rank"] = rank

# report the top-ranked entities extracted from this text chunk

df: pd.DataFrame = pd.DataFrame([
    node_attr
    for node, node_attr in graph.nodes(data = True)
    if node_attr["kind"] == "Entity" and node_attr["text"] not in STOP_WORDS
]).sort_values(by = [ "rank", "count" ], ascending = False)

print(df.head(20))


# use `pyvis` to provide an interactive visualization of both layers of the graph

pv_net: pyvis.network.Network = pyvis.network.Network(
    notebook = False,
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
pv_net.save_graph("chunk.html")

