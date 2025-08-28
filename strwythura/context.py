#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the domain context, using `RDFlib` and related libraries.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import defaultdict
import json
import pathlib
import typing

from rdflib.namespace import DCTERMS, RDF, SKOS
import gensim  # type: ignore
import networkx as nx
import rdflib

from .graph import Entity


class DomainContext:
    """
Represent the domain context using an _ontology pipeline_ process:
vocabulary, taxonomy, thesaurus, and ontology.
    """
    IRI_BASE: str = "https://github.com/DerwenAI/strwythura/#"
    IRI_PREFIX: str = "strw:"
    LEMMA_PHRASE: rdflib.term.URIRef = rdflib.term.URIRef(f"{IRI_BASE}lemma_phrase")


    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = {}
        self.rdf_graph: rdflib.Graph = rdflib.Graph()
        self.w2v_vectors: list = []
        self.w2v_model: typing.Optional[ gensim.models.Word2Vec ] = None
        self.known_lemma: typing.List[ str ] = []
        self.taxo_node: typing.Dict[ str, int ] = {}
        self.sem_layer: nx.MultiDiGraph = nx.MultiDiGraph()


    def set_config (
        self,
        config: dict,
        ) -> None:
        """
Accessor method to configure -- part of a design pattern to make the
domain context handling more "pluggable", i.e., to be subclassed and
customized for other use cases.
        """
        self.config = config

        # load the RDF-based context for the domain
        domain_path: pathlib.Path = pathlib.Path(self.config["kg"]["domain_path"])

        self.rdf_graph.parse(
            domain_path.as_posix(),
            format = "turtle",
        )


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


    def get_first_lemma (
        self,
        concept_iri: rdflib.term.Node,
        ) -> str:
        """
Get the primary lemma for a `SKOS:Concept` entity.
        """
        return next(
            self.rdf_graph.objects(concept_iri, self.LEMMA_PHRASE)
        ).toPython()  # type: ignore


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


    def abbrev_concept (
        self,
        concept_iri: rdflib.term.Node,
        ) -> str:
        """
Abbreviate a `SKOS:Concept` entity's IRI with the vocabulary prefix.
        """
        return concept_iri.toPython().replace(self.IRI_BASE, self.IRI_PREFIX)  # type: ignore


    def populate_taxonomy_node (
        self,
        concept_iri: rdflib.term.URIRef,
        ) -> typing.Tuple[ int, str, dict ]:
        """
Get the attributes for a `SKOS:Concept` entity.
        """
        lemmas: typing.List[ str ] = [
            lemma.toPython()  # type: ignore
            for lemma in self.rdf_graph.objects(concept_iri, self.LEMMA_PHRASE)
        ]

        lemma_key: str = lemmas[0]
        self.add_lemma(lemma_key)

        node_id: int = self.get_lemma_index(lemma_key)
        label: str = self.abbrev_concept(concept_iri)
        self.taxo_node[label] = node_id

        self.sem_layer.add_node(
            node_id,
            kind = "Taxonomy",
            key = lemma_key,
            label = label,
            text = self.rdf_graph.value(
                concept_iri,
                SKOS.definition,
            ).toPython(),  # type: ignore
            iri = self.rdf_graph.value(
                concept_iri,
                DCTERMS.identifier,
            ).toPython(),  # type: ignore
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
        node_map: typing.Dict[ str, int ] = {}
        attr_map: typing.Dict[ int, dict ] = {}

        # first pass: populate nodes for the `SKOS:Concept` entities
        for concept_iri in self.rdf_graph.subjects(RDF.type, SKOS.Concept):
            node_id, lemma_key, attr = self.populate_taxonomy_node(concept_iri)  # type: ignore
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
                        key = rel_iri,
                        prob = 1.0,
                    )


    def get_ner_labels (
        self,
        ) -> typing.List[ str ]:
        """
Iterate through `SKOS:Concept` entities to extract the labels used for
zero-shot NER.
        """
        return [
            label.toPython()  # type: ignore
            for concept_iri in self.rdf_graph.subjects(RDF.type, SKOS.Concept)
            for label in self.rdf_graph.objects(concept_iri, SKOS.prefLabel, unique = True)
        ]


    def get_label_map (
        self,
        ) -> typing.Dict[ str, str ]:
        """
Iterate through `SKOS:Concept` entities to extract a mapping between
NER labels and abbreviated IRIs.
        """
        return {
            label.toPython(): self.abbrev_concept(concept_iri)  # type: ignore
            for concept_iri in self.rdf_graph.subjects(RDF.type, SKOS.Concept)
            for label in self.rdf_graph.objects(concept_iri, SKOS.prefLabel, unique = True)
        }


    def add_w2v_vectors (
        self,
        span_decoder: typing.Dict[ tuple, Entity ],
        ) -> None:
        """
Build the vector input for entity embeddings.
        """
        w2v_map: typing.Dict[ int, typing.Set[ str ]] = defaultdict(set)

        for ent in span_decoder.values():
            if ent.node is not None:
                w2v_map[ent.sent_id].add(ent.key)

        for sent_id, ents in w2v_map.items():
            vec: list = list(ents)
            vec.insert(0, str(sent_id))
            self.w2v_vectors.append(vec)


    def embed_entities (
        self,
        *,
        w2v_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Train a `gensim.Word2Vec` model for entity embeddings.
        """
        w2v_max: int = max([  # pylint: disable=R1728
            len(vec) - 1
            for vec in self.w2v_vectors
        ])

        self.w2v_model = gensim.models.Word2Vec(
            self.w2v_vectors,
            min_count = 2,
            window = w2v_max,
        )

        if w2v_path is None:
            w2v_path = pathlib.Path(self.config["ent"]["w2v_path"])

        self.w2v_model.save(w2v_path.as_posix())


    def save_sem_layer (
        self,
        *,
        kg_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Serialize the KG
        """
        if kg_path is None:
            kg_path = pathlib.Path(self.config["kg"]["kg_path"])

        with kg_path.open("w", encoding = "utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(
                        self.sem_layer,
                        edges = "edges",
                    ),
                    indent = 2,
                    sort_keys = True,
                )
            )


    def serialize_assets (
        self,
        *,
        kg_path: typing.Optional[ pathlib.Path ] = None,
        w2v_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Serialize the assets for reusing a constructed KG.
        """
        self.embed_entities(w2v_path = w2v_path)
        self.save_sem_layer(kg_path = kg_path)
