#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the domain context, using `RDFlib` and related libraries.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import defaultdict
import json
import math
import pathlib
import typing

from icecream import ic  # type: ignore
from rdflib.namespace import DCTERMS, RDF, ORG, SKOS
import gensim  # type: ignore
import lancedb  # type: ignore
import networkx as nx
import rdflib
import spacy

from .elem import Entity, NodeKind, StrwVocab, TextChunk


class DomainContext:  # pylint: disable=R0902,R0904
    """
Represent the domain context using an _ontology pipeline_ process:
vocabulary, taxonomy, thesaurus, and ontology.
    """
    STRW_PREFIX: str = "strw:"
    STRW_BASE: str = "https://github.com/DerwenAI/strwythura/#"

    SZ_PREFIX: str = "sz:"

    POS_TRANSFORM: typing.Dict[ str, str ] = {
        "PROPN": "NOUN",
    }


    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = {}
        self.rdf_graph: rdflib.Graph = rdflib.Graph()

        self.start_chunk_id: int = 0
        self.chunk_table: typing.Optional[ lancedb.table.LanceTable ] = None

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


    def init_chunk_table (
        self,
        ) -> None:
        """
Initialize the chunk table in the vector store.
        """
        vect_db: lancedb.db.LanceDBConnection = lancedb.connect(self.config["vect"]["lancedb_uri"])  # pylint: disable=C0301

        self.chunk_table = vect_db.create_table(
            self.config["vect"]["chunk_table"],
            schema = TextChunk,
            mode = "overwrite",
        )

        self.start_chunk_id = 0


    def parse_lemma (
        self,
        span: list,
        *,
        debug: bool = False,
        ) -> str:
        """
Construct a parsed, lemmatized key for the given noun phrase.
        """
        lemmas: typing.List[ str ] = []

        for tok in span:
            pos: str = tok.pos_
            lemma: str = tok.lemma_.strip().lower()

            if pos in self.POS_TRANSFORM:
                pos = self.POS_TRANSFORM[pos]

            lemmas.append(f"{pos}.{lemma}")

        lemma_key: str = " ".join(lemmas)

        if debug:
            ic(lemma_key, span)

        return lemma_key


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
Get the primary lemma for a `skos:Concept` entity.
        """
        lemma_phrase_iri: rdflib.term.URIRef = self.rel_iri(StrwVocab.LEMMA_PHRASE)

        return next(
            self.rdf_graph.objects(concept_iri, lemma_phrase_iri)
        ).toPython()  # type: ignore


    def form_concept (
        self,
        fragment: str,
        ) -> rdflib.term.URIRef:
        """
Lookup a `skos:Concept` entity by its IRI.
        """
        iri: str = f"{self.STRW_BASE}{fragment}"
        concept_iri: rdflib.term.URIRef = rdflib.term.URIRef(iri)

        return concept_iri


    def rel_iri (
        self,
        rel: StrwVocab,
        ) -> rdflib.term.URIRef:
        """
Accessor to construct a `URIRef` for a relation within the `strw:` vocabulary.
        """
        return rdflib.term.URIRef(
            self.STRW_BASE + rel.value.replace(self.STRW_PREFIX, "")
        )


    def populate_taxo_node (
        self,
        concept_iri: rdflib.term.URIRef,
        ) -> typing.Tuple[ int, str, dict ]:
        """
Populate a semantic layer node from a `skos:Concept` entity in the taxonomy.
        """
        lemma_phrase_iri: rdflib.term.URIRef = self.rel_iri(StrwVocab.LEMMA_PHRASE)

        lemmas: typing.List[ str ] = [
            lemma.toPython()  # type: ignore
            for lemma in self.rdf_graph.objects(concept_iri, lemma_phrase_iri)
        ]

        lemma_key: str = lemmas[0]
        self.add_lemma(lemma_key)

        node_id: int = self.get_lemma_index(lemma_key)
        label: str = concept_iri.n3(self.rdf_graph.namespace_manager)
        self.taxo_node[label] = node_id

        text: str = self.rdf_graph.value(
            concept_iri,
            SKOS.definition,
        ).toPython()  # type: ignore

        iri: str = self.rdf_graph.value(
            concept_iri,
            DCTERMS.identifier,
        ).toPython()  # type: ignore

        self.chunk_table.add([  # type: ignore
            TextChunk(
                uid = self.start_chunk_id,
                url = iri,
                sent_id = 0,
                text = text,
            )
        ])

        self.start_chunk_id += 1

        self.sem_layer.add_node(
            node_id,
            kind = NodeKind.TAXONOMY.value,
            key = lemma_key,
            label = label,
            text = text,
            iri = iri,
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
Iterate through `skos:Concept` entities, loading into `NetworkX`
        """
        node_map: typing.Dict[ str, int ] = {}
        attr_map: typing.Dict[ int, dict ] = {}

        # first pass: populate nodes for the `skos:Concept` entities
        for concept_iri in self.rdf_graph.subjects(RDF.type, SKOS.Concept):
            node_id, lemma_key, attr = self.populate_taxo_node(concept_iri)  # type: ignore
            node_map[lemma_key] = node_id
            attr_map[node_id] = attr

        # second pass: add relations
        for src_id, attr in attr_map.items():
            for rel in [ "broader", "narrower", "related" ]:
                rel_iri: str = f"skos:{rel}"

                for dst_key in attr[rel]:
                    dst_id: int = node_map[dst_key]

                    self.sem_layer.add_edge(
                        src_id,
                        dst_id,
                        key = rel_iri,
                        prob = 1.0,
                    )


    def parse_er_export (  # pylint: disable=R0912,R0913,R0914,R0915
        self,
        datasets: typing.List[ str ],
        *,
        export_path: typing.Optional[ pathlib.Path ] = None,
        rdf_path: typing.Optional[ pathlib.Path ] = None,
        language: str = "en",
        debug: bool = True,
        ) -> None:
        """
Parse the Senzing entity resolution results exported as JSON.
        """
        if export_path is None:
            export_path = pathlib.Path(self.config["er"]["export_path"])

        if rdf_path is None:
            rdf_path = pathlib.Path(self.config["er"]["thesaurus_path"])

        rdf_list: typing.List[ str ] = [
            """
@prefix strw:  <https://github.com/DerwenAI/strwythura/#> .
@prefix sz:    <https://senzing.com/#> .

@prefix dct:   <http://purl.org/dc/terms/> .
@prefix org:   <http://www.w3.org/ns/org#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix skos:  <http://www.w3.org/2004/02/skos/core#> .
            """
        ]

        org_map: typing.Dict[ str, str ] = {}
        parent: typing.Dict[ str, str ] = {}

        simple_pipe: spacy.Language = spacy.load(self.config["nlp"]["spacy_model"])

        # load the data records
        data_records: typing.Dict[ str, dict ] = {}

        for filename in datasets:
            data_path: pathlib.Path = pathlib.Path(filename)

            with open(data_path, encoding = "utf-8") as fp:
                for line in fp:
                    rec: dict = json.loads(line)
                    record_id: str = self.SZ_PREFIX + rec["DATA_SOURCE"].replace(" ", "_").lower() + "_" + rec["RECORD_ID"]  # pylint: disable=C0301
                    data_records[record_id] = rec

        # parse the JSON export
        with open(export_path, encoding = "utf-8") as fp:
            for line in fp:
                data: dict = json.loads(line)

                entity_id: str = self.SZ_PREFIX + str(data["RESOLVED_ENTITY"]["ENTITY_ID"])
                ent_descrip: str = ""
                ent_type: str = ""

                rec_list: typing.List[ dict ] = []
                rel_list: typing.List[ dict ] = []

                for rec in data["RESOLVED_ENTITY"]["RECORDS"]:
                    ent_descrip = rec["ENTITY_DESC"]

                    record_id = rec["RECORD_ID"]
                    data_source: str = rec["DATA_SOURCE"].replace(" ", "_").lower()
                    rec_iri: str = f"{self.SZ_PREFIX}{data_source}_{record_id}"
                    parent[rec_iri] = entity_id

                    pred_iri: str = "skos:exactMatch"

                    rec_list.append({
                        "pred": pred_iri,
                        "obj": rec_iri,
                        "skos:prefLabel": rec["ENTITY_DESC"],
                    })

                for rel in data["RELATED_ENTITIES"]:
                    match_key: str = rel["MATCH_KEY"]
                    match_level: int = rel["MATCH_LEVEL"]
                    match_code: str = rel["MATCH_LEVEL_CODE"]

                    why: str = f"{match_key} {match_level}"
                    pred_iri = "skos:related"

                    if match_code == "POSSIBLY_SAME":
                        pred_iri = "skos:closeMatch"

                    rel_list.append({
                        "pred": pred_iri,
                        "obj": self.SZ_PREFIX + str(rel["ENTITY_ID"]),
                        "skos:definition": why,
                    })

                ent_node: dict = {
                    "iri": entity_id,
                    "skos:prefLabel": ent_descrip,
                }

                if debug:
                    ic(ent_node)

                rdf_frag: str = f"{entity_id} skos:prefLabel \"{ent_descrip}\"@{language} "

                lemma_key: str = self.parse_lemma(simple_pipe(ent_descrip))  # type: ignore
                rdf_frag += f";\n  strw:lemma_phrase \"{lemma_key}\"@{language} "

                for rec_node in rec_list:
                    dat_rec: dict = data_records[rec_node["obj"]]
                    ent_type = dat_rec["RECORD_TYPE"]
                    rdf_frag += f';\n  {rec_node["pred"]} {rec_node["obj"]} '

                    if ent_type == "ORGANIZATION":
                        org_map[rec_node["skos:prefLabel"]] = entity_id

                for rel_node in rel_list:
                    rdf_frag += f';\n  {rel_node["pred"]} {rel_node["obj"]} '

                rdf_frag += f";\n  rdf:type strw:SzEntity, strw:{ent_type.capitalize()} "
                rdf_frag += "\n."
                rdf_list.append(rdf_frag)

        # construct the RDF graph
        for record_id, rec in data_records.items():
            rec_type: str = f'strw:{rec["RECORD_TYPE"].capitalize()}'
            name: str = ""
            employer: str = ""
            urls: typing.List[ str ] = []

            if rec_type == "strw:Organization":
                name = rec["NAMES"][0]["PRIMARY_NAME_ORG"]

                if "LINKS" in rec:
                    for url_dict in rec["LINKS"]:
                        for url in url_dict.values():
                            urls.append(url)

                if "WEBSITE_ADDRESS" in rec:
                    urls.append(rec["WEBSITE_ADDRESS"])

            else:
                if "NAME_FIRST" in rec:
                    name = rec["NAME_FIRST"]

                if "NAME_MIDDLE" in rec:
                    name += " " + rec["NAME_MIDDLE"]

                if "NAME_LAST" in rec:
                    name += " " + rec["NAME_LAST"]

                if "SOURCE_LINKS" in rec:
                    for url_dict in rec["SOURCE_LINKS"]:
                        for url in url_dict.values():
                            urls.append(url)

                if "EMPLOYER_NAME" in rec:
                    org_name: str = rec["EMPLOYER_NAME"]

                    if org_name in org_map:
                        employer = org_map.get(org_name)  # type: ignore

            rdf_frag = f"{record_id} rdf:type strw:DataRecord, {rec_type} "
            rdf_frag += f";\n  skos:prefLabel \"{name}\"@{language} "

            lemma_key = self.parse_lemma(simple_pipe(name))  # type: ignore
            rdf_frag += f";\n  strw:lemma_phrase \"{lemma_key}\"@{language} "

            for url in urls:
                rdf_frag += f";\n  dct:identifier <{url}> "

            rdf_frag += "\n."
            rdf_list.append(rdf_frag)

            if len(employer) > 0:
                rdf_frag = f"{parent[record_id]} org:memberOf {employer} ."
                rdf_list.append(rdf_frag)

        # serialize the generated RDF file
        with open(rdf_path, "w", encoding = "utf-8") as fp:
            fp.write("\n".join(rdf_list))

        # load the RDF graph
        rdf_graph: rdflib.Graph = rdflib.Graph()
        rdf_graph.parse(
            rdf_path.as_posix(),
            format = "turtle",
        )


    def populate_er_node (
        self,
        er_graph: rdflib.Graph,
        entity_iri: rdflib.term.Node,
        ) -> int:
        """
Populate a semantic layer node from an ER entity.
        """
        lemma_phrase_iri: rdflib.term.URIRef = self.rel_iri(StrwVocab.LEMMA_PHRASE)

        lemmas: typing.List[ str ] = [
            lemma.toPython()  # type: ignore
            for lemma in er_graph.objects(entity_iri, lemma_phrase_iri)
        ]

        lemma_key: str = lemmas[0]
        self.add_lemma(lemma_key)
        node_id: int = self.get_lemma_index(lemma_key)
        label: str = entity_iri.n3(er_graph.namespace_manager)

        text: str = er_graph.value(
            entity_iri,
            SKOS.prefLabel,
        ).toPython()  # type: ignore

        self.sem_layer.add_node(
            node_id,
            kind = NodeKind.ENTITY.value,
            key = lemma_key,
            label = label,
            text = text,
            iri = entity_iri,
            rank = 0.0,
            count = 1,
        )

        return node_id


    def load_er_thesaurus (  # pylint: disable=R0914
        self,
        datasets: typing.List[ str ],
        er_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Iterate through the _entity resolution_ results, adding a
domain-specific thesaurus of entities and relations into the
semantic layer.
        """
        self.parse_er_export(datasets)

        node_map: typing.Dict[ str, int ] = {}

        # load the ER triples into their own graph, to extrant and
        # link the known lemmas (i.e., the synonyms in the thesaurus)
        if er_path is None:
            er_path = pathlib.Path(self.config["er"]["thesaurus_path"])

        er_graph: rdflib.Graph = rdflib.Graph()

        er_graph.parse(
            er_path.as_posix(),
            format = "turtle",
        )

        # first iterate through the data records, loading lemma keys
        # and populating nodes in the semantic layer
        for entity_iri in er_graph.subjects(RDF.type, self.form_concept("DataRecord")):
            node_id = self.populate_er_node(er_graph, entity_iri)
            node_map[entity_iri.n3(er_graph.namespace_manager)] = node_id

        # now iterate through the entities, overriding any prior lemma
        # keys from data records
        for entity_iri in er_graph.subjects(RDF.type, self.form_concept("SzEntity")):
            node_id = self.populate_er_node(er_graph, entity_iri)
            node_map[entity_iri.n3(er_graph.namespace_manager)] = node_id

        # then add SKOS relations (thesaurus synonyms and taxonymy)
        # as edges in the semantic layer
        for entity_iri in er_graph.subjects(RDF.type, self.form_concept("SzEntity")):
            for sem_rel in [ SKOS.related, SKOS.closeMatch, SKOS.exactMatch, ORG.memberOf ]:
                for obj in er_graph.objects(entity_iri, sem_rel):
                    src_id: int = node_map[entity_iri.n3(er_graph.namespace_manager)]
                    dst_id: int = node_map[obj.n3(er_graph.namespace_manager)]

                    if src_id != dst_id:
                        rel_iri: str = sem_rel.n3(er_graph.namespace_manager)
                        prob: float = 0.5

                        if rel_iri in [ "skos:exactMatch", "org:memberOf" ]:
                            prob = 1.0

                        self.sem_layer.add_edge(
                            src_id,
                            dst_id,
                            key = rel_iri,
                            prob = prob,
                        )

        # also link entities to their taxonomy nodes
        for taxo_iri in [ self.form_concept("Organization"), self.form_concept("Person") ]:
            for entity_iri in er_graph.subjects(RDF.type, taxo_iri):
                node_id = node_map[entity_iri.n3(er_graph.namespace_manager)]
                taxo_node_id: int = self.taxo_node[taxo_iri.n3(self.rdf_graph.namespace_manager)]

                self.sem_layer.add_edge(
                    node_id,
                    taxo_node_id,
                    key = "RDF:type",
                    weight = 0.0
                )

        # finally, load ER triples into the semantic layer
        self.rdf_graph.parse(
            er_path.as_posix(),
            format = "turtle",
        )


    def get_ner_labels (
        self,
        ) -> typing.List[ str ]:
        """
Iterate through `skos:Concept` entities to extract the labels used for
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
Iterate through `skos:Concept` entities to extract a mapping between
NER labels and abbreviated IRIs.
        """
        return {
            label.toPython(): concept_iri.n3(self.rdf_graph.namespace_manager)  # type: ignore
            for concept_iri in self.rdf_graph.subjects(RDF.type, SKOS.Concept)
            for label in self.rdf_graph.objects(concept_iri, SKOS.prefLabel, unique = True)
        }


    def vis_nodes (
        self,
        num_docs: int,
        ) -> typing.Iterator[ typing.Tuple[ int, dict ]]:
        """
Iterator for the visualization attributes of nodes.
        """
        for node_id, node_attr in self.sem_layer.nodes(data = True):
            attr: dict = {}

            if node_attr.get("kind") == NodeKind.ENTITY.value and node_attr.get("label") not in [ "NP" ]:  # pylint: disable=C0301
                attr["color"] = "hsla(65, 46%, 58%, 0.80)"
                attr["size"] = round(20 * math.log(1.0 + math.sqrt(float(node_attr.get("count"))) / num_docs))  # type: ignore # pylint: disable=C0301
                attr["label"] = node_attr.get("text")  # type: ignore
                attr["title"] = node_attr.get("key")  # type: ignore
            elif node_attr.get("kind") == NodeKind.TAXONOMY.value:
                attr["color"] = "hsla(306, 45%, 57%, 0.95)"
                attr["size"] = 5
                attr["label"] = node_attr.get("label")  # type: ignore
                attr["title"] = node_attr.get("iri")  # type: ignore
            else:
                continue

            yield node_id, attr


    def vis_edges (
        self,
        ) -> typing.Iterator[ typing.Tuple[ int, int, str ]]:
        """
Iterator for the visualization attributes of edges.
        """
        for src_node, dst_node, key in self.sem_layer.edges(keys = True):
            yield src_node, dst_node, key


    def add_entity_sequence (
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
Serialize the constructed KG as a JSON file represented in the
_node-link_ data format.

Aternatively this could be stored in a graph database.
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


    def save_assets (
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


    def load_assets (
        self,
        *,
        kg_path: typing.Optional[ pathlib.Path ] = None,
        w2v_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Load the serialized assets for a previoulys constructed KG.
        """
        vect_db: lancedb.db.LanceDBConnection = lancedb.connect(self.config["vect"]["lancedb_uri"])
        self.chunk_table = vect_db.open_table(self.config["vect"]["chunk_table"])

        if w2v_path is None:
            w2v_path = pathlib.Path(self.config["ent"]["w2v_path"])

        self.w2v_model = gensim.models.Word2Vec.load(w2v_path.as_posix())

        if kg_path is None:
            kg_path = pathlib.Path(self.config["kg"]["kg_path"])

        with pathlib.Path(kg_path).open("r", encoding = "utf-8") as fp:
            self.sem_layer = nx.node_link_graph(
                json.load(fp),
                edges = "edges",
            )
