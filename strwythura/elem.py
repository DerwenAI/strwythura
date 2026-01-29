#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Elements used for knowledge graph construction.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from enum import StrEnum
import typing

from pydantic import BaseModel


STRW_BASE: str = "https://github.com/DerwenAI/strwythura/wiki/vocab#"
STRW_PREFIX: str = "strw:"


class NodeKind (StrEnum):
    """
Values for the `kind` property in graph nodes.
    """
    CHUNK = "Chunk"
    ENTITY = "Entity"
    DATAREC = "DataRec"
    TAXONOMY = "Taxonomy"


class EntitySource (StrEnum):
    """
Values for the `source` property in `Entity` instances.
    """
    TAXO = "Domain_Taxonomy"
    ER = "Entity_Resolution"
    NER = "Named_Entity_Recognition"
    NC = "Noun_Chunk"
    LEX = "Parsed_Noun"


class EntityInstance (BaseModel):
    """
Represents coordinates for one instance of an entity among the chunks.
    """
    sent_id: int
    chunk_id: int


class NounSpan (BaseModel):
    """
Represents one token span within a parsed sentence.
    """
    loc: tuple[ int, int ]
    text: str
    span: list
    label: str | None = None
    source: EntitySource = EntitySource.LEX
    iri: str | None = None


class Entity (BaseModel):
    """
Represents one entity instance:

  * defined by the domain taxonomy
  * determined by entity resolution from structured data sources
  * extracted from unstructured data sources

A non-null `iri` field indicates this entity is linked within
the constructed knowledge graph.
    """
    span: NounSpan
    lemma_key: str
    uid: int | None = None
    inst: list[ EntityInstance ] = []
    count: int = 0
    rank: float = 0.0

    def get_iri (
        self,
        ) -> str:
        """
Construct an IRI based on the `lemma_key` value.
        """
        if self.span.source == EntitySource.ER:
            return self.span.iri  # type: ignore

        stub: str = self.lemma_key.replace(" ", "_")
        return f"{STRW_PREFIX}lemma_{stub}"


def de_token_span (
    span: typing.Any,
    ) -> list[ str ]:
    """
Convert from a `spaCy` token span to a list of strings.
    """
    return list(map(str, span))


class NodeStyle (BaseModel):
    """
Represent graph visualization styles for nodes.
    """
    color: str
    shape: str = "dot"


NODE_STYLES: dict[ EntitySource, NodeStyle ] = {
    EntitySource.TAXO: NodeStyle(
        color = "hsla(306, 45%, 57%, 0.5)",
        shape = "box",
    ),

    EntitySource.ER: NodeStyle(
        color = "hsl(55, 17%, 49%, 0.9)",
    ),

    EntitySource.NER: NodeStyle(
        color = "hsla(65, 46%, 58%, 0.8)",
    ),
}
