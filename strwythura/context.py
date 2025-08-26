#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the domain context, using `RDFlib` and related libraries.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import pathlib
import typing

from rdflib.namespace import RDF, SKOS
import rdflib


class DomainContext:  # pylint: disable=R0902,R0903
    """
Represent the domain context using an _ontology pipeline_ process:
vocabulary, taxonomy, thesaurus, and ontology.
    """

    def __init__ (
        self,
        config: dict,
        ) -> None:
        """
Constructor.
        """
        # configuration
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
Extract the labels used for zero-shot NER and corresponding graph
nodes within the semantic layer definition for this domain context.
        """
        return [
            str(label)
            for concept in self.rdf_graph.subjects(RDF.type, SKOS.Concept)
            for label in self.rdf_graph.objects(concept, SKOS.prefLabel, unique = True)
        ]
