#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the domain context.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import Counter, defaultdict, OrderedDict
import inspect
import itertools
import json
import pathlib
import sys
import typing

from icecream import ic
from lancedb.embeddings import get_registry
from lancedb.embeddings.sentence_transformers import SentenceTransformerEmbeddings
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


EMBED_FUNC: SentenceTransformerEmbeddings = \
    get_registry().get("sentence-transformers").create()


class TextChunk (LanceModel):
    """
Represents one chunk of text from a document.
    """
    uid: int
    url: str
    sent_id: int
    text: str = EMBED_FUNC.SourceField()
    vector: Vector(EMBED_FUNC.ndims()) = EMBED_FUNC.VectorField(default = None)  # type: ignore

    @classmethod
    def get_iri (
        cls,
        uid: int,
        ) -> str:
        """
Construct an IRI based on the chunk `uid` value.
        """
        return f"{STRW_PREFIX}chunk_{uid}"


    @classmethod
    def get_uid (
        cls,
        iri: int,
        ) -> str:
        """
Extract the `uid` value based on a chunk IRI.
        """
        stub: str = f"{STRW_PREFIX}chunk_"
        return int(iri.replace(stub, ""))


class DomainContext:
    """
Represent the domain context using an _ontology pipeline_ process:
vocabulary, taxonomy, thesaurus, and ontology.
    """
    TAXO_SENT_ID: int = 0


    def __init__ (
        self,
        config: dict,
        thesaurus: Thesaurus,
        ent_store: EntityStore,
        lex: LexicalGraph,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config

        # entities and semantic layer
        self.thesaurus: Thesaurus = thesaurus
        self.ent_store: EntityStore = ent_store

        # intermediate parsing outcomes
        self.lex: LexicalGraph = lex

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
            uids: list[ int ] = [ uid for uid in df_chunks.iter_rows() ]

            if len(uids) > 0:
                self.start_chunk_id = max(uids)[0] + 1
            else:
                self.start_chunk_id = 0


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
        self.add_node(
            TextChunk.get_iri(chunk.uid),
            NodeKind.CHUNK,
            attrs = {
                "chunk" : chunk.uid,
                "url": chunk.url,
                "sent": chunk.sent_id,
            },
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
SELECT DISTINCT ?concept_iri ?label
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


    def promote_data_nodes (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Reform selected RDF triples in `RDFlib` to be represented in a `NetworkX`
property graph: for the provenance of data records used in ER.
        """
        query: str = """
SELECT DISTINCT ?rec_iri ?rec_key ?data_src
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
            self.add_node(
                rec_iri,
                NodeKind.DATAREC,
                attrs = {
                    "rec_key": rec_key,
                    "data_src": data_src,
                },
            )


    def promote_taxo_nodes (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Reform selected RDF triples in `RDFlib` to be represented in a `NetworkX`
property graph: for the `SKOS:Concept` items from the taxonomy.

Also add embeddings for each `SKOS:definition` text in the vector store.
        """
        query: str = """
SELECT DISTINCT ?concept_iri ?text ?lemma
WHERE {
  ?concept_iri a skos:Concept ;
    skos:definition ?text ;
    sz:lemma_phrase ?lemma ;
  .
}""".strip()

        qres: SPARQLResult = self.thesaurus.rdf_graph.query(query)

        # iterate through the SPARQL query results
        for i, row in enumerate(qres):
            concept_iri: str = self.thesaurus.n3(row[0])
            text: str = row[1].toPython()
            lemma_key: str = row[2].toPython()

            if debug:
                ic(i, concept_iri, text, lemma_key)

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
                self.TAXO_SENT_ID, # zero sentence reserved for taxonomy concepts
                text,
            )

            # add node to the ERKG
            self.add_node(
                concept_iri,
                NodeKind.TAXONOMY,
                attrs = {
                    "count": found_ent.count,
                    "rank": found_ent.rank,
                    "text": text,
                    "lemma": lemma_key,
                    "method": EntitySource.TAXO.value,
                },
                stop = False,
            )

        # query for SKOS relations within the taxonomy,
        # then add ERKG edges to represent these
        query = """
SELECT DISTINCT ?ent ?sem_rel ?rel
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
            self.add_edge(
                ent_iri,
                sem_rel,
                rel_iri,
	        prob = 1.0,
                update = True,
            )


    def promote_er_nodes (
        self,
        parser: "Parser",
        *,
        debug: bool = False,
        ) -> None:
        """
Reform selected RDF triples in `RDFlib` to be represented in a `NetworkX`
property graph: for the entity definitions from ER.
        """
        query: str = """
SELECT DISTINCT ?ent ?ent_class ?label
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

            ## TODO: THIS LOGIC IS HORKED!!
            ## each ER needs to become a distinct entity
            if len(label) < 1:
                # create a ERKG node, though without an entity definition
                uid: int = self.ent_store.increment_uid()
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
                        iri = ent_iri,
                    ),
                    lemma_key = lemma_key,
                )

                found_ent: Entity = self.ent_store.encode_entity(
                    ent,
                    create = True,
                )

                rank = found_ent.rank

            # add node to the ERKG
            self.add_node(
                ent_iri,
                NodeKind.ENTITY,
                attrs = {
                    "count": found_ent.count,
                    "rank": rank,
                    "text": label,
                    "lemma": lemma_key,
                    "method": EntitySource.ER.value,
                }
            )

            # add a ERKG edge to link to the SKOS:concept class
            self.add_edge(
	        ent_iri,
                self.thesaurus.n3(RDF.type),
                concept_iri,
	        prob = 1.0,
                update = True,
            )


    def promote_er_edges (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Reform selected RDF triples in `RDFlib` to be represented in a `NetworkX`
property graph: for the SKOS relations from ER.
        """
        # query blank nodes for ent => ent | rec
        # SKOS relations, then add edges
        query: str = """
SELECT DISTINCT ?ent ?rel_ent ?sem_rel ?key ?lev
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
            rel_iri: str = self.thesaurus.n3(row[1])

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
            self.add_edge(
	        ent_iri,
                sem_rel,
                rel_iri,
	        prob = prob,
                attrs = {
                    "match_key": match_key,
                    "match_level": match_level,
                },
                update = True,
            )


    def promote_ner_nodes (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Reform selected RDF triples in `RDFlib` to be represented in a `NetworkX`
property graph: for the entities extracted from NER.
        """
        # iterate through the entity store
        for ent in self.ent_store.entities.values():
            if ent.span.source >= EntitySource.NER:
                # add node to the ERKG
                self.add_node(
                    ent.get_iri(),
                    NodeKind.ENTITY,
                    attrs = {
                        "count": ent.count,
                        "rank": ent.rank,
                        "text": ent.span.text,
                        "lemma": ent.lemma_key,
                        "method": ent.span.source.value,
                    },
                    update = True,
                )

                # add a ERKG edge to link to the SKOS:concept class
                self.add_edge(
	            ent.get_iri(),
                    self.thesaurus.n3(RDF.type),
                    ent.span.iri,
	            prob = 1.0,
                    update = True,
                )


    ######################################################################
    ## manage additional entity context

    def link_entity_chunks (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Cross-link entities with the chunks in which they appear.
        """
        sem_rel: str = f"{STRW_PREFIX}within_chunk"

        for ent in self.ent_store.entities.values():
            for ent_inst in ent.inst:
                self.add_edge(
                    ent.get_iri(),
                    sem_rel,
                    TextChunk.get_iri(ent_inst.chunk_id),
                    prob = 1.0,
                    attrs = {
                        "weight": ent.rank,
                        "sent": ent_inst.sent_id,
                    },
                    update = True,
                )


    def co_occur_entities (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Connect entities which co-occur within the same sentence.
        """
        inst_dict: dict[ int, dict[ int, int ]] = defaultdict(lambda: defaultdict(list))
        counter: Counter = Counter() 

        # partition entity co-occurrence by `( chunk_id, sent_id, ent.uid, )`
        for ent in self.ent_store.entities.values():
            if ent.span.source >= EntitySource.NER:
                for ent_inst in ent.inst:
                    inst_dict[ent_inst.chunk_id][ent_inst.sent_id].append(ent.uid)

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
        decoder: dict[ int, Entity ] = self.ent_store.get_decoder()
        sem_rel: str = f"{STRW_PREFIX}co_occurs_with"

        for pair, count in counter.items():
            tally[pair[0]].append(( pair, count, ))

        for elem, pairs_list in tally.items():
            partition: Counter = Counter(dict(pairs_list))
            total: float = float(partition.total())

            for pair, count in partition.items():
                prob: float = round(float(count) / total, 4)

                if debug:
                    ic(pair, count, prob)

                # add relation into the ERKG
                self.add_edge(
                    decoder[pair[0]].get_iri(),
                    sem_rel,
                    decoder[pair[1]].get_iri(),
                    prob = prob,
                    update = True,
                )


    ######################################################################
    ## manage the knowledge graph

    def add_edge (
        self,
        src_iri: str,
        rel_iri: str,
        dst_iri: str,
        prob = 0.0,
        *,
        attrs: dict = {},
        update: bool = False,
        stop: bool = True,
        debug: bool = False,
        ) -> dict | None:
        """
Add an edge into the ERKG with required and optional properties.

Required properties: each edge must have an IRI as its `MultiGraph`
unique key, and a `prob` probability value.

Optional properties: specified as key/value pairs in the `attrs`
dictionary.
        """
        edge: tuple = ( src_iri, dst_iri, rel_iri, )
        pre_exist: bool = False

        # override conflicting settings
        if update:
            stop = False

        # test whether the edge IRI already exists in the ERKG?
        if self.erkg.has_edge(*edge):
            pre_exist = True

            calframe: list = inspect.getouterframes(inspect.currentframe(), 2)
            caller: str = calframe[1][3]
            prev_attrs: dict = self.erkg.edges[*edge]

            if debug | stop:
                print(f"dupe: {caller} {edge} {prob} {attrs}")
                print("PRE-EXISTING EDGE", prev_attrs)

            if stop:
                # if requested for debugging, stop the application
                sys.exit(-1)
            elif not update:
                # return the pre-existing edge data and do not update
                return prev_attrs

        # add an edge into the ERKG
        if not pre_exist:
            if debug:
                ic("ADD EDGE", edge, prob, attrs)

            self.erkg.add_edge(
	        src_iri,
                dst_iri,
	        key = rel_iri,
	        prob = prob,
            )

        # set the optional edge attributes, if any
        if (update or not pre_exist) and len(attrs) > 0:
            nx.set_edge_attributes(
                self.erkg,
                { edge: attrs },
            )

        return None


    def add_node (
        self,
        iri: str,
        kind: NodeKind,
        *,
        attrs: dict = {},
        update: bool = False,
        stop: bool = True,
        debug: bool = False,
        ) -> dict | None:
        """
Add a node into the ERKG with required and optional properties.

Required properties: each node must have an IRI as its unique
identifier, and a `NodeKind` value.

Optional properties: specified as key/value pairs in the `attrs`
dictionary.
        """
        pre_exist: bool = False

        # override conflicting settings
        if update:
            stop = False

        # test whether the node IRI already exists in the ERKG?
        if self.erkg.has_node(iri):
            pre_exist = True

            calframe: list = inspect.getouterframes(inspect.currentframe(), 2)
            caller: str = calframe[1][3]
            prev_attrs: dict = self.erkg.nodes[iri]

            if debug | stop:
                print(f"dupe: {caller} {iri} {kind}")
                print("PRE-EXISTING NODE", prev_attrs)

            if stop:
                # if requested for debugging, stop the application
                sys.exit(-1)
            elif not update:
                # return the pre-existing node data and do not update
                return prev_attrs

        # add a node into the ERKG
        if not pre_exist:
            if debug:
                ic("ADD NODE", iri, kind.value, attrs)

            self.erkg.add_node(
                iri,
                kind = kind.value,
            )
        else:
            attrs["kind"] = kind.value

        # set the optional node attributes, if any
        if (update or not pre_exist) and len(attrs) > 0:

            nx.set_node_attributes(
                self.erkg,
                { iri: attrs },
            )

        return None


    def get_node (
        self,
        iri: str,
        ) -> dict:
        """
Accessor method to get the properties of an ERKG node.
        """
        return self.erkg.nodes[iri]


    def load_graph (
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


    def save_graph (
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
