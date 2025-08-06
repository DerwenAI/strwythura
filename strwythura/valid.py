#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data validation classes.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from dataclasses import dataclass
import typing

from lancedb.embeddings import get_registry, transformers
from lancedb.pydantic import LanceModel, Vector
import spacy

EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

EMBED_FCN: transformers.TransformersEmbeddingFunction = \
    get_registry().get("huggingface").create(name = EMBED_MODEL)


class TextChunk (LanceModel):
    uid: int
    url: str
    sent_id: int
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
