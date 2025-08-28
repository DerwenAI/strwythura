#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data validation classes for constructing knowledge graphs.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from dataclasses import dataclass
from enum import StrEnum
import typing

from lancedb.embeddings import get_registry, transformers  # type: ignore
from lancedb.pydantic import LanceModel, Vector  # type: ignore
import spacy


class NodeKind (StrEnum):
    """
Values for the `kind` property in graph nodes.
    """
    CHUNK = "Chunk"
    ENTITY = "Entity"
    TAXONOMY = "Taxonomy"


class StrwVocab (StrEnum):
    """
Values for relations in the `strw:` RDF vocabulary.
    """
    LEMMA_PHRASE = "strw:lemma_phrase"
    FOLLOWS_LEXICALLY = "strw:follows_lexically"
    CO_OCCURS_WITH = "strw:co_occurs_with"
    COMPOUND_ELEM_OF = "strw:compound_elem_of"
    WITHIN_CHUNK = "strw:within_chunk"


# Note: `LanceDB` requires that the embedding model be hard-coded, so far
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
