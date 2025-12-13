#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the domain context.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import Counter, defaultdict, OrderedDict
import itertools
import json
import pathlib
import sys
import typing

from icecream import ic
from lancedb.embeddings import get_registry, transformers
from lancedb.pydantic import LanceModel, Vector
from rdflib import Namespace
from rdflib.namespace import RDF
from rdflib.plugins.sparql.processor import SPARQLResult
from sz_semantics import Thesaurus
import lancedb
import networkx as nx
import polars as pl
import spacy

from .elem import Entity, EntitySource, NodeKind, NounSpan, \
    STRW_PREFIX
from .ent import EntityStore
from .lex import LexicalGraph


# NB: `LanceDB` requires the embedding model to be hard-coded (so far)
EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

EMBED_FUNC: transformers.TransformersEmbeddingFunction = \
    get_registry().get("huggingface").create(name = EMBED_MODEL)


class TextChunk (LanceModel):
    """
Represents one chunk of text from a document.
    """
    uid: int
    url: str
    sent_id: int
    text: str = EMBED_FUNC.SourceField()
    vector: Vector(EMBED_FUNC.ndims()) = EMBED_FUNC.VectorField(default = None)


class DomainContext:
    """
Represent the domain context using an _ontology pipeline_ process:
vocabulary, taxonomy, thesaurus, and ontology.
    """

    def __init__ (
        self,
        config: dict,
        thesaurus: Thesaurus,
        ent_store: EntityStore,
        lexical: LexicalGraph,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config

        # entities and semantic layer
        self.thesaurus: Thesaurus = thesaurus
        self.ent_store: EntityStore = ent_store

        # some entities and data records won't have lemma, so we'll
        # provide an alternative means of node lookup based on IRI
        self.iri_map: dict[ str, str ] = {}

        # intermediate parsing outcomes
        self.lexical: LexicalGraph = lexical

        # the constructed knowledge graph in `NetworkX`
        # each edge has: `src_id`, `dst_id`, `key` (relation), `prob`
        self.erkg: nx.MultiDiGraph = nx.MultiDiGraph()

        # the vector store in `LanceDB`
        self.lancedb_conn: lancedb.db.LanceDBConnection = lancedb.connect(
            self.config["vect"]["lancedb_uri"],
        )

        self.start_chunk_id: int = 0
        self.chunk_table: lancedb.table.LanceTable | None = None


    ######################################################################
    ## manage the vector store

    def open_vector_tables (
        self,
        *,
        create: bool = False,
        ) -> None:
        """
Open the table for text chunk embeddings in the vector store,
overwriting any previous data if indicated.
        """
        if create:
            # intialize and clear any previous table
            self.chunk_table = self.lancedb_conn.create_table(
                self.config["vect"]["chunk_table"],
                schema = TextChunk,
                mode = "overwrite",
            )
        else:
            # open existing table
            self.chunk_table = self.lancedb_conn.open_table(
                self.config["vect"]["chunk_table"],
            )

            df_chunks: pl.DataFrame = self.chunk_table.search().select([ "uid" ]).to_polars()
            self.start_chunk_id = max([ uid for uid in df_chunks.iter_rows() ])[0]


    def add_chunk (
        self,
        url: str,
        sent_id: int,
        text: str,
        ) -> TextChunk:
        """
Add a chunk into both the vector store and the ERKG.
        """
        chunk: TextChunk = TextChunk(
            uid = self.start_chunk_id,
            url = url,
            sent_id = sent_id,
            text = text,
        )

        ## add to the vector store
        self.chunk_table.add([ chunk ])
        self.start_chunk_id += 1

        # add node to the ERKG
        chunk_node_id: str = f"chunk_{chunk.uid}"

        self.test_node(
            chunk_node_id,
            chunk.url,
            "add_chunk",
        )

        self.erkg.add_node(
            chunk_node_id,
            kind = NodeKind.CHUNK.value,
            chunk = chunk.uid,
            url = chunk.url,
            rank = 0.0,
        )

        return chunk


    def get_chunk_meta (
        self,
        ) -> typing.Iterator[ tuple[ int, str ]]:
        """
Iterator for TextChunk metadata from the `LanceDB` table.
        """
        for uid, url in self.chunk_table.search().select([ "uid", "url" ]).to_polars().iter_rows():
            yield uid, url


    ######################################################################
    ## manage the semantics

    def get_label_map (
        self,
        ) -> dict[ str, str ]:
        """
Accessor: iterate through `skos:Concept` entities to extract a mapping
between NER labels and abbreviated IRIs.

Used for _entity linking_ in using zero-shot NER tasks, such as the `GLiNER`
library.
        """
        query: str = """
SELECT ?concept_iri ?label
WHERE {
  ?concept_iri a skos:Concept ;
    skos:prefLabel ?label ;
    sz:ner_label true ;
  .
}""".strip()

        qres: SPARQLResult = self.thesaurus.rdf_graph.query(query)

        return {
            label.toPython(): self.thesaurus.n3(concept_iri)
            for concept_iri, label in qres
        }


    def promote_taxo_nodes (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Reform the semantic graph in `RDFlib` => property graph in `NetworkX`
for the `SKOS:Concept` items from the taxonomy.

Also add embeddings for each `SKOS:definition` text in the vector store.
        """
        query: str = """
SELECT ?concept_iri ?text ?label ?lemma
WHERE {
  ?concept_iri a skos:Concept ;
    skos:definition ?text ;
    skos:prefLabel ?label ;
    sz:lemma_phrase ?lemma ;
  .
}""".strip()

        qres: SPARQLResult = self.thesaurus.rdf_graph.query(query)

        # iterate through the SPARQL query results
        for row in qres:
            concept_iri: str = self.thesaurus.n3(row[0])
            text: str = row[1].toPython()
            label: str = row[2].toPython()
            lemma_key: str = row[3].toPython()

            if debug:
                ic(concept_iri, text, label, lemma_key)

            # create an entry in the entity store
            ent: Entity = Entity(
                span = NounSpan(
                    loc = ( -1, -1, ),
                    text = "",
                    span = [],
                    source = EntitySource.TAXO,
                ),
                lemma_key = lemma_key,
            )

            found_ent: Entity = self.ent_store.encode_entity(
                ent,
                create = True,
            )

            # add the text and its embedding for the `SKOS:definition`
            # as a chunk in the vector store
            chunk_id: int = self.add_chunk(
                concept_iri,
                0, # zero sentence is reserved for taxonomy concepts
                text,
            )

            # add node to the ERKG
            self.iri_map[concept_iri] = found_ent.node_id

            self.test_node(
                found_ent.node_id,
                concept_iri,
                "promote_taxo_nodes",
            )

            self.erkg.add_node(
                found_ent.node_id,
                kind = NodeKind.TAXONOMY.value,
                label = label,
                lemma = lemma_key,
                text = text,
                iri = concept_iri,
                rank = found_ent.rank,
                count = 0,
            )

        # query for SKOS relations within the taxonomy,
        # then add ERKG edges to represent these
        query = """
SELECT ?ent ?sem_rel ?rel
WHERE {
  VALUES ?sem_rel {
    skos:broader
    skos:narrower
    skos:related
  } .
  ?ent a skos:Concept ;
    ?sem_rel ?rel .
}""".strip()

        qres = self.thesaurus.rdf_graph.query(query)

        # iterate through the SPARQL query results
        for row in qres:
            ent_iri: str = self.thesaurus.n3(row[0])
            sem_rel: str = self.thesaurus.n3(row[1])
            rel_iri: str = self.thesaurus.n3(row[2])

            if debug:
                ic(ent_iri, sem_rel, rel_iri)

            # add a ERKG edge for the related entities
            self.erkg.add_edge(
                self.iri_map[ent_iri],
                self.iri_map[rel_iri],
	        key = sem_rel,
	        prob = 1.0,
            )


    def promote_data_nodes (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Reform the semantic graph in `RDFlib` => property graph in `NetworkX`
to represent the data record provenance from ER.
        """
        query: str = """
SELECT ?rec_iri ?rec_key ?data_src
WHERE {
  ?rec_iri a sz:DataRecord ;
    dc:identifier ?rec_key ;
    prov:wasQuotedFrom ?data_src ;
  .
}""".strip()

        # iterate through the SPARQL query results
        qres: SPARQLResult = self.thesaurus.rdf_graph.query(query)

        for row in qres:
            rec_iri: str = self.thesaurus.n3(row[0])
            rec_key: str = row[1].toPython().strip()
            data_src: str = self.thesaurus.n3(row[2])

            if debug:
                ic(rec_iri, rec_key, data_src)

            # add node to the ERKG
            node_id: int = self.ent_store.increment_nodes()
            self.iri_map[rec_iri] = node_id

            self.test_node(
                node_id,
                rec_iri,
                "promote_data_nodes",
            )

            self.erkg.add_node(
                node_id,
                kind = NodeKind.DATAREC.value,
                label = rec_iri,
                text = rec_iri,
                iri = rec_iri,
                rec_key = rec_key,
                data_src = data_src,
                rank = 0.0,
                count = 0,
            )


    def promote_er_nodes (
        self,
        parser: "Parser",
        *,
        debug: bool = False,
        ) -> None:
        """
Reform the semantic graph in `RDFlib` => property graph in `NetworkX`
to represent the entity definitions from ER.
        """
        query: str = """
SELECT ?ent ?ent_class ?label
WHERE {
  VALUES ?ent_class {
    sz:Person
    sz:Organization
  } .
  ?ent a ?ent_class ;
    skos:prefLabel ?label ;
  .
}""".strip()

        # iterate through the SPARQL query results
        qres: SPARQLResult = self.thesaurus.rdf_graph.query(query)

        for row in qres:
            ent_iri: str = self.thesaurus.n3(row[0])
            concept_iri: str = self.thesaurus.n3(row[1])
            label: str = row[2].toPython().strip()
            rank: float = 0.0

            if debug:
                ic(ent_iri, concept_iri, label)

            if len(label) < 1:
                # create a ERKG node, though without an entity definition
                node_id: int = self.ent_store.increment_nodes()
                lemma_key: str = ""
                label = ent_iri

            else:
                # use the label to generate a lemma key
                span: spacy.tokens.doc.Doc = parser.ner_pipe(label)
                lemma_key = parser.tokenize_lemma(span)

                # create an entry in the entity store
                ent: Entity = Entity(
                    span = NounSpan(
                        loc = ( 0, len(span) - 1, ),
                        text = label,
                        span = span,
                        source = EntitySource.ER,
                    ),
                    lemma_key = lemma_key,
                )

                found_ent: Entity = self.ent_store.encode_entity(
                    ent,
                    create = True,
                )

                node_id = found_ent.node_id
                rank = found_ent.rank

            # add node to the ERKG
            self.iri_map[ent_iri] = node_id

            self.test_node(
                node_id,
                ent_iri,
                "promote_er_nodes",
            )

            self.erkg.add_node(
                node_id,
                kind = NodeKind.ENTITY.value,
                label = label,
                lemma = lemma_key,
                text = label,
                iri = ent_iri,
                rank = rank,
                count = 0,
            )

            # add a ERKG edge to link to the SKOS:concept class
            self.erkg.add_edge(
	        node_id,
                self.iri_map[concept_iri],
	        key = RDF.type,
	        prob = 1.0,
            )


    def promote_er_edges (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Reform the semantic graph in `RDFlib` => property graph in `NetworkX`
to represent the SKOS relations from ER.
        """
        # query blank nodes for ent => ent | rec
        # SKOS relations, then add edges
        query: str = """
SELECT ?ent ?rel_ent ?sem_rel ?key ?lev
WHERE {
  ?bl rdf:predicate ?sem_rel ;
    rdf:subject ?ent ;
    rdf:object ?rel_ent ;
    sz:match_key ?key ;
    sz:match_level ?lev ;
  .
}""".strip()

        # iterate through the SPARQL query results
        qres: SPARQLResult = self.thesaurus.rdf_graph.query(query)

        for row in qres:
            ent_iri: str = self.thesaurus.n3(row[0])
            ent_node_id: int = self.iri_map[ent_iri]

            rel_iri: str = self.thesaurus.n3(row[1])
            rel_node_id: int = self.iri_map[rel_iri]

            sem_rel: str = self.thesaurus.n3(row[2])
            prob: float = 1.0

            match_key: str = row[3].toPython().strip()
            match_level: int = row[4].toPython()

            match match_level:
                case 11:
                    # Senzing calls this a `disclosed relationship`
                    # which alternatively might be `sz:member_of`
                    prob = 1.0
                case 2:
                    prob = 0.8
                case 3:
                    prob = 0.5

            if debug:
                ic(ent_iri, rel_iri, sem_rel, match_key, match_level)

            # add a ERKG edge for ent => ent | rec relations
            self.erkg.add_edge(
	        ent_node_id,
                rel_node_id,
	        key = sem_rel,
	        prob = prob,
                match_key = match_key,
                match_level = match_level,
            )


    def promote_ner_nodes (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Reform the semantic graph in `RDFlib` => property graph in `NetworkX`
for the entities extracted from NER.
        """
        # iterate through the entity store
        for ent in self.ent_store.entities.values():
            if ent.span.source >= EntitySource.NER:
                if ent.span.label is None:
                    # create a default label for noun chunks
                    label: str = "NC"
                else:
                    label: str = ent.span.label

                # add node to the ERKG
                self.test_node(
                    ent.node_id,
                    ent.lemma_key,
                    "promote_ner_nodes",
                )

                self.erkg.add_node(
                    ent.node_id,
                    kind = NodeKind.ENTITY.value,
                    label = label,
                    lemma = ent.lemma_key,
                    source = ent.span.source.value,
                    text = ent.span.text,
                    iri = ent.span.iri,
                    rank = ent.rank,
                    count = ent.count,
                )


    ######################################################################
    ## entity co-occurrence

    def co_occur_entities (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Connect entities which co-occur within the same sentence.
        """
        sem_rel: str = f"{STRW_PREFIX}co_occurs_with"
        inst_dict: dict[ int, dict[ int, int ]] = defaultdict(lambda: defaultdict(list))
        counter: Counter = Counter() 

        # partition entity co-occurrence by `( chunk_id, sent_id, node_id, )`
        for ent in self.ent_store.entities.values():
            for ent_inst in ent.inst:
                inst_dict[ent_inst.chunk_id][ent_inst.sent_id].append(ent.node_id)

        if debug:
            ic(inst_dict)

        # tally the pairwise co-occurrence of entities
        for chunk_id, sent_dict in sorted(inst_dict.items()):
            for sent_id, ent_list in sorted(sent_dict.items()):
                for pair in itertools.combinations(ent_list, 2):
                    pair: tuple = tuple(sorted(pair))
                    counter[pair] += 1

                    pair = tuple(sorted(pair, reverse = True))
                    counter[pair] += 1

        if debug:
            ic(counter)

        # partition by first element, to compute a conditional
        # probability per second element
        tally: dict[ int, list ] = defaultdict(list)

        for pair, count in counter.items():
            tally[pair[0]].append(( pair, count, ))

        for elem, pairs_list in tally.items():
            partition: Counter = Counter(dict(pairs_list))
            total: float = float(partition.total())

            for pair, count in partition.items():
                prob: float = round(float(count) / total, 4)

                if debug:
                    ic(pair, count, prob)

                # add relation into the lexical graph
                self.lexical.lex_graph.add_edge(
                    pair[0],
                    pair[1],
                    key = sem_rel,
                    prob = prob,
                )


    ######################################################################
    ## manage the knowledge graph

    def test_node (
        self,
        node_id: int | str,
        label: str,
        message: str,
        *,
        full_stop: bool = False,
        ) -> None:
        """
Verify that a `node_id` does not already exist in the ERKG.
        """
        if not self.erkg.has_node(node_id):
            return

        print(f"{message}: {node_id} {label}")
        ic(self.erkg.nodes[node_id])

        if full_stop:
            sys.exit(-1)


    def load_erkg (
        self,
        erkg_path: pathlib.Path,
        ) -> None:
        """
De-serialize a constructed KG from a JSON file represented in the
_node-link_ data format.
        """
        with erkg_path.open("r", encoding = "utf-8") as fp:
            self.erkg = nx.node_link_graph(
                json.load(fp),
                edges = "edges",
            )


    def save_erkg (
        self,
        erkg_path: pathlib.Path,
        ) -> None:
        """
Serialize the constructed KG as a JSON file represented in the
_node-link_ data format.

Aternatively this could be stored in a graph database.
        """
        with erkg_path.open("w", encoding = "utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(
                        self.erkg,
                        edges = "edges",
                    ),
                    indent = 2,
                    sort_keys = True,
                )
            )
