#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the domain context, using `RDFlib` and related libraries.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import pathlib
import typing

from rdflib.namespace import DCTERMS, RDF, SKOS
import networkx as nx
import rdflib


class DomainContext:  # pylint: disable=R0902,R0903
    """
Represent the domain context using an _ontology pipeline_ process:
vocabulary, taxonomy, thesaurus, and ontology.
    """
    IRI_BASE: str = "https://github.com/DerwenAI/strwythura/#"
    LEMMA_PHRASE: rdflib.term.URIRef = rdflib.term.URIRef(f"{IRI_BASE}lemma_phrase")


    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = {}
        self.known_lemma: typing.List[ str ] = []
        self.sem_layer: nx.Graph = nx.Graph()


    def set_config (
        self,
        config: dict,
        ) -> None:
        """
Accessor method to configure -- part of a design pattern to make the
domain context handling more "pluggable", i.e., to be subclassed and
customized for other use cases.
        """
        self.config: dict = config

        # load the RDF-based context for the domain
        domain_path: pathlib.Path = pathlib.Path(self.config["kg"]["domain_path"])
        self.rdf_graph: rdflib.Graph = rdflib.Graph()

        self.rdf_graph.parse(
            domain_path.as_posix(),
            format = "turtle",
        )


    def get_ner_labels (
        self,
        ) -> typing.List[ str ]:
        """
Iterate through `SKOS:Concept` entities to extract the labels used for
zero-shot NER.
        """
        return [
            str(label)
            for concept_iri in self.rdf_graph.subjects(RDF.type, SKOS.Concept)
            for label in self.rdf_graph.objects(concept_iri, SKOS.prefLabel, unique = True)
        ]


    def get_lemma_index (
        self,
        lemma_key: str,
        ) -> int:
        """
Lookup the UID for nodes in the semantic layer, based on a parsed
lemma key for a known entity.
        """
        return self.known_lemma.index(lemma_key)


    def add_lemma (
        self,
        lemma_key: str,
        ) -> bool:
        """
Add a known entity, indexed by its parsed lemma key.
        """
        prev_known: bool = True

        if lemma_key not in self.known_lemma:
            self.known_lemma.append(lemma_key)
            prev_known = False

        return prev_known


    def lookup_concept (
        self,
        fragment: str,
        ) -> rdflib.term.URIRef:
        """
Lookup a `SKOS:Concept` entity by its IRI.
        """
        iri: str = f"{self.IRI_BASE}{fragment}"
        concept_iri: rdflib.term.URIRef = rdflib.term.URIRef(iri)

        return concept_iri


    def get_first_lemma (
        self,
        concept_iri: rdflib.term.URIRef,
        ) -> str:
        """
Get the primary lemma for a `SKOS:Concept` entity.
        """
        return next(self.rdf_graph.objects(concept_iri, self.LEMMA_PHRASE)).toPython()


    def populate_taxonomy_node (
        self,
        concept_iri: rdflib.term.URIRef,
        ) -> typing.Tuple[ int, str, dict ]:
        """
Get the attributes for a `SKOS:Concept` entity.
        """
        lemmas: typing.List[ str ] = [
            lemma.toPython()
            for lemma in self.rdf_graph.objects(concept_iri, self.LEMMA_PHRASE)
        ]

        lemma_key: str = lemmas[0]
        self.add_lemma(lemma_key)

        node_id: int = self.get_lemma_index(lemma_key)

        self.sem_layer.add_node(
            node_id,
            kind = "Entity",
            key = lemma_key,
            text = self.rdf_graph.value(concept_iri, SKOS.definition).toPython(),
            label = next(self.rdf_graph.objects(concept_iri, SKOS.prefLabel, unique = True)).toPython(),
            iri = self.rdf_graph.value(concept_iri, DCTERMS.identifier).toPython(),
            rank = 0.0,
            count = 0,
        )

        # scheduled as relations to get added, once the nodes are in place
        attrs = {
            "lemmas": lemmas,
            "broader": [
                self.get_first_lemma(node)
                for node in self.rdf_graph.objects(concept_iri, SKOS.broader)
            ],
            "narrower": [
                self.get_first_lemma(node)
                for node in self.rdf_graph.objects(concept_iri, SKOS.narrower)
            ],
            "related": [
                self.get_first_lemma(node)
                for node in self.rdf_graph.objects(concept_iri, SKOS.related)
            ],
        }

        return node_id, lemma_key, attrs


    def load_taxonomy (
        self,
        ) -> None:
        """
Iterate through `SKOS:Concept` entities, loading into `NetworkX`
        """
        node_map: typing.Dict[ rdflib.term.URIRef, int ] = {}
        attr_map: typing.Dict[ int, dict ] = {}

        # first pass: populate nodes for the `SKOS:Concept` entities
        for concept_iri in self.rdf_graph.subjects(RDF.type, SKOS.Concept):
            node_id, lemma_key, attr = self.populate_taxonomy_node(concept_iri)
            node_map[lemma_key] = node_id
            attr_map[node_id] = attr

        # second pass: add relations
        for src_id, attr in attr_map.items():
            for rel in [ "broader", "narrower", "related" ]:
                rel_iri: str = f"SKOS:{rel}"

                for dst_key in attr[rel]:
                    dst_id: int = node_map[dst_key]

                    self.sem_layer.add_edge(
                        src_id,
                        dst_id,
                        rel = rel_iri,
                    )
