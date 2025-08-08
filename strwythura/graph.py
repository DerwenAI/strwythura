#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data validation classes.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from dataclasses import dataclass
import typing

from lancedb.embeddings import get_registry, transformers  # type: ignore
from lancedb.pydantic import LanceModel, Vector  # type: ignore
import spacy


# Note: this model is hard-coded, so far
EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

EMBED_FCN: transformers.TransformersEmbeddingFunction = \
    get_registry().get("huggingface").create(name = EMBED_MODEL)


class TextChunk (LanceModel):
    """
Represents one chunk of text from a document.
    """
    uid: int
    url: str
    sent_id: int
    text: str = EMBED_FCN.SourceField()
    vector: Vector(EMBED_FCN.ndims()) = EMBED_FCN.VectorField(default = None)  # type: ignore # pylint: disable=E1136


@dataclass(order=False, frozen=False)
class Entity:  # pylint: disable=R0902
    """
Represents one entity in the graph.
    """
    loc: typing.Tuple[ int, int ]
    key: str
    text: str
    label: str
    chunk_id: int
    sent_id: int
    span: spacy.tokens.span.Span  # pylint: disable=I1101
    node: typing.Optional[ int ] = None
