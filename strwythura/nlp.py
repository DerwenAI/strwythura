#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NLP utilities.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import logging
import os
import typing
import unicodedata

from gliner_spacy.pipeline import GlinerSpacy
import glirel
import spacy
import transformers

from .graph import TextChunk


CHUNK_SIZE: int = 1024
GLINER_MODEL: str = "urchade/gliner_small-v2.1"

NER_LABELS: typing.List[ str ] = [
    "Behavior",
    "City",
    "Company",    
    "Condition",
    "Conference",
    "Country",
    "Food",
    "Food Additive",
    "Hospital",
    "Organ",
    "Organization",
    "People Group",
    "Person",
    "Publication",
    "Research",
    "Science",
    "University",
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

STOP_WORDS: typing.Set[ str ] = set([
    "PRON.it",
    "PRON.that",
    "PRON.they",
    "PRON.those",
    "PRON.we",
    "PRON.which",
    "PRON.who",
])


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


def init_nlp_pipe (
    config: dict,
    ) -> spacy.Language:
    """
Initialize the models.
    """
    # override specific Hugging Face error messages, since
    # `transformers` and `tokenizers` have noisy logging
    logging.disable(logging.ERROR)
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "0"

    # load models for `spaCy`, `GLiNER`, `GLiREL`
    # this may take several minutes when run the first time
    nlp_pipe: spacy.Language = spacy.load(config["nlp"]["spacy_model"])

    nlp_pipe.add_pipe(
        "gliner_spacy",
        config = {
            "gliner_model": GLINER_MODEL,
            "labels": NER_LABELS,
            "chunk_size": CHUNK_SIZE,
            "style": "ent",
        },
    )
        
    nlp_pipe.add_pipe(
        "glirel",
        after = "ner",
    )

    return nlp_pipe


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
