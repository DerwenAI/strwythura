#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NLP methods for constructing the _lexical graph_.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing
import unicodedata

from gliner_spacy.pipeline import GlinerSpacy  # type: ignore # pylint: disable=W0611
from icecream import ic  # type: ignore
import networkx as nx
import spacy
import w3lib.html

from .context import DomainContext
from .elem import StrwVocab, TextChunk


class Parser:
    """
Wrapper class for the `spaCy` NLP pipeline used to extract entities
and relations, based on using `GLiNER`, BAML, and textgraphs.
    """
    STOP_WORDS: typing.Set[ str ] = set([
        "PRON.each",
        "PRON.he",
        "PRON.it",
        "PRON.she",
        "PRON.some",
        "PRON.someone",
        "PRON.that",
        "PRON.their",
        "PRON.they",
        "PRON.those",
        "PRON.we",
        "PRON.what",
        "PRON.which",
        "PRON.who",
        "PRON.you",
    ])


    def __init__ (
        self,
        config: dict,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config
        self.ner_labels: typing.List[ str ] = []


    def build_entity_pipe (
        self,
        ner_labels: typing.List[ str ],
        ) -> spacy.Language:
        """
Initialize the `spaCy` pipeline used for NER + RE, by loading models
for `spaCy`, `GLiNER`

  - `ner_labels`: semantics to apply for zero-shot NER

Note: this may take several minutes when run the first time after
installing the repo.
        """
        self.ner_labels = ner_labels
        entity_pipe: spacy.Language = spacy.load(self.config["nlp"]["spacy_model"])

        entity_pipe.add_pipe(
            "gliner_spacy",
            config = {
                "style": "ent",
                "labels": self.ner_labels,
                "gliner_model": self.config["nlp"]["gliner_model"],
                "chunk_size": self.config["vect"]["chunk_size"],
            },
        )

        return entity_pipe


    def parse_text (  # pylint: disable=R0913,R0914
        self,
        domain_context: DomainContext,
        entity_pipe: spacy.Language,
        lex_graph: nx.MultiDiGraph,
        chunk: TextChunk,
        *,
        debug: bool = False,  # pylint: disable=W0613
        ) -> spacy.tokens.doc.Doc:  # pylint: disable=I1101
        """
Parse an input text chunk, returning a `spaCy` document.
        """
        doc: spacy.tokens.doc.Doc = entity_pipe(chunk.text)  # type: ignore  # pylint: disable=I1101

        # scan the document tokens to add lemmas into the _lexical graph_
        # then use a _textgraph_ distillation approach called "textrank"
        for sent in doc.sents:
            node_seq: typing.List[ int ] = []

            if False: # debug  # pylint: disable=W0125
                ic(sent)

            for tok in sent:
                text: str = tok.text.strip()
                pos: str = tok.pos_

                if pos in DomainContext.POS_TRANSFORM:
                    pos = DomainContext.POS_TRANSFORM[pos]

                if pos == "NOUN":
                    lemma_key: str = domain_context.parse_lemma([ tok ])  # type: ignore
                    prev_known: bool = domain_context.add_lemma(lemma_key)
                    node_id: int = domain_context.get_lemma_index(lemma_key)
                    node_seq.append(node_id)

                    if not lex_graph.has_node(node_id):
                        lex_graph.add_node(
                            node_id,
                            key = lemma_key,
                            kind = "Lemma",
                            pos = pos,
                            text = text,
                            chunk = chunk,
                            count = 1,
                        )

                    elif prev_known:
                        node: dict = lex_graph.nodes[node_id]
                        node["count"] += 1

            # create the _textrank_ edges for the lexical graph,
            # which will get used for ranking, but discarded later
            if False: # debug  # pylint: disable=W0125
                ic(node_seq)

            for hop in range(self.config["tr"]["tr_lookback"]):
                for node_id, node in enumerate(node_seq[: -1 - hop]):  # type: ignore
                    neighbor: int = node_seq[hop + node_id + 1]

                    if not lex_graph.has_edge(node, neighbor):
                        lex_graph.add_edge(
                            node,
                            neighbor,
                            key = StrwVocab.FOLLOWS_LEXICALLY.value,
                        )

        return doc


    def uni_scrubber (
        self,
        span: spacy.tokens.span.Span,  # pylint: disable=I1101
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
        limpio = w3lib.html.replace_escape_chars(limpio)

        limpio = limpio.replace('“', '"').replace('”', '"')
        limpio = limpio.replace("‘", "'").replace("’", "'").replace("`", "'").replace("â", "'")
        limpio = limpio.replace("…", "...").replace("–", "-")

        limpio = str(unicodedata.normalize("NFKD", limpio).encode("ascii", "ignore").decode("utf-8"))  # pylint: disable=C0301

        return limpio


    def make_chunk (
        self,
        doc: spacy.tokens.doc.Doc,  # pylint: disable=I1101
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
        sent_id: int = 0

        for sent_id, sent in enumerate(doc.sents):
            line: str = self.uni_scrubber(sent)
            line_len: int = len(line)

            if (chunk_total + line_len) > self.config["vect"]["chunk_size"]:
                # emit the current chunk
                chunk_list.append(
                    TextChunk(
                        uid = chunk_id,
                        url = url,
                        sent_id = sent_id,
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
                sent_id = sent_id + 1,
                text = "\n".join(chunks),
            )
        )

        return chunk_id + 1
