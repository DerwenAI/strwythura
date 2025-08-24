#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds assets for constructing a KG, then running GraphRAG downstream.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import itertools
import json
import logging
import os
import pathlib
import tomllib
import traceback
import typing
import warnings

from icecream import ic  # type: ignore
from rdflib.namespace import RDF, SKOS
import gensim  # type: ignore
import lancedb  # type: ignore
import networkx as nx
import polars as pl
import rdflib
import spacy
import transformers

from .baml_client import b
from .baml_client import types as baml_types
from .graph import TextChunk
from .kg import KnowledgeGraph
from .nlp import Parser
from .vis import gen_pyvis


class Strwythura:  # pylint: disable=R0902
    """
Builds assets for constructing a KG, then running GraphRAG downstream.
    """

    def __init__ (
        self,
        *,
        config_path: pathlib.Path = pathlib.Path("config.toml"),
        ) -> None:
        """
Constructor.
        """
        # configuration
        self.config: dict = {}

        with open(config_path, mode = "rb") as fp:
            self.config = tomllib.load(fp)

        # disable noisy logging
        os.environ["BAML_LOG"] = "WARN"
        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        logging.disable(logging.ERROR)
        transformers.logging.set_verbosity_error()

        ## none of this works!
        #os.environ["TQDM_DISABLE"] = "1"
        #loguru.logger.disable(gliner_spacy.pipeline.__name__)
        #loggers: dict =
        # { name:logging.getLogger(name) for name in logging.root.manager.loggerDict }
        #ic(loggers)

        # load the semantic layer: define a context for the domain
        domain_path: pathlib.Path = pathlib.Path(self.config["kg"]["domain_path"])
        self.domain_graph: rdflib.Graph = rdflib.Graph()

        self.domain_graph.parse(
            domain_path.as_posix(),
            format = "turtle",
        )

        # initialize the data structures used for assets
        self.parser: Parser = Parser(self.config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.simple_pipe: spacy.Language = spacy.load(self.config["nlp"]["spacy_model"])
            self.entity_pipe: typing.Optional[ spacy.Language ] = None
            self.chunk_table: typing.Optional[ lancedb.table.LanceTable ] = None
            self.sem_overlay: nx.Graph = nx.Graph()
            self.w2v_vectors: list = []
            self.w2v_model: typing.Optional[ gensim.models.Word2Vec ] = None


    def get_ner_labels (
        self,
        ) -> typing.List[ str ]:
        """
Extract the labels used for zero-shot NER and corresponding graph
nodes within the semantic layer definition for this domain context.
        """
        return [
            str(label)
            for concept in self.domain_graph.subjects(RDF.type, SKOS.Concept)
            for label in self.domain_graph.objects(concept, SKOS.prefLabel, unique = True)
        ]


    def build_assets (  # pylint: disable=R0913
        self,
        url_list: typing.List[ str ],
        *,
        debug: bool = False,
        kg_path: typing.Optional[ pathlib.Path ] = None,
        w2v_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Builds assets for constructing a KG.
        """
        self.parser.update_data(
            url_list,
            self.get_ner_labels(),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                # given the NER labels, build the `spaCy` pipe
                self.entity_pipe = self.parser.build_entity_pipe()

                # initialize the chunk table
                vect_db: lancedb.db.LanceDBConnection = lancedb.connect(self.config["vect"]["lancedb_uri"])  # pylint: disable=C0301

                self.chunk_table = vect_db.create_table(
                    self.config["vect"]["chunk_table"],
                    schema = TextChunk,
                    mode = "overwrite",
                )

                # construct the graph
                kg: KnowledgeGraph = KnowledgeGraph(self.config)

                kg.build_graph(
                    self.parser,
                    self.simple_pipe,
                    self.entity_pipe,
                    self.chunk_table,
                    self.sem_overlay,
                    self.w2v_vectors,
                    debug = debug,
                )

                # serialize assets
                self.embed_entities(w2v_path = w2v_path)
                self.save_graph(kg_path = kg_path)

            except Exception as ex:  # pylint: disable=W0718
                ic(ex)
                traceback.print_exc()


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


    def save_graph (
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
                        self.sem_overlay,
                        edges = "edges",
                    ),
                    indent = 2,
                    sort_keys = True,
                )
            )


    def gen_visualization (
        self,
        *,
        html_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Generate HTML for an interactive visualization of the graph, based on `PyVis`
        """
        if html_path is None:
            html_path = pathlib.Path(self.config["kg"]["html_path"])

        gen_pyvis(
            self.sem_overlay,
            html_path.as_posix(),
            num_docs = len(self.parser.url_list),
        )


    def load_assets (
        self,
        *,
        kg_path: typing.Optional[ pathlib.Path ] = None,
        w2v_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Load the serialized assets for a constructed KG.
        """
        vect_db: lancedb.db.LanceDBConnection = lancedb.connect(self.config["vect"]["lancedb_uri"])
        self.chunk_table = vect_db.open_table(self.config["vect"]["chunk_table"])

        if w2v_path is None:
            w2v_path = pathlib.Path(self.config["ent"]["w2v_path"])

        self.w2v_model = gensim.models.Word2Vec.load(w2v_path.as_posix())

        if kg_path is None:
            kg_path = pathlib.Path(self.config["kg"]["kg_path"])

        with pathlib.Path(kg_path).open("r", encoding = "utf-8") as fp:
            self.sem_overlay = nx.node_link_graph(
                json.load(fp),
                edges = "edges",
            )

        # build the `spaCy` pipe, no need for input URL list
        self.parser.update_data(
            [],
            self.get_ner_labels(),
        )

        self.entity_pipe = self.parser.build_entity_pipe()


class GraphRAG:
    """
Run an example query through LanceDB to identify _chunks_ and through
the Word2Vec entity embedding model for a _semantic expansion_ to
produce a set of _anchor nodes_ in the NetworkX graph.
    """

    def __init__ (
        self,
        strw: Strwythura,
        ) -> None:
        """
Constructor.
        """
        self.strw: Strwythura = strw


    def find_entities (
        self,
        question: str,
        ) -> typing.Iterator[ str ]:
        """
Extract entity spans from a text question.
        """
        doc: spacy.tokens.doc.Doc = self.strw.entity_pipe(question)  # type: ignore  # pylint: disable=I1101

        for span in doc.ents:
            key: str = " ".join([
                tok.pos_ + "." + tok.lemma_.strip().lower()
                for tok in span
            ])

            yield key


    def get_chunks (  # pylint: disable=R0914
        self,
        question: str,
        *,
        debug: bool = False,
        num_chunks: typing.Optional[ int ] = None,
        ) -> typing.List[ str ]:
        """
Run semantic search to produce a set of text chunks.
        """
        if num_chunks is None:
            num_chunks = self.strw.config["rag"]["num_chunks"]

        # extract entities from the question
        entities: typing.Set[ str ] = set(list(self.find_entities(question)))

        if debug:
            ic(entities)

        # semantic expansion using entity embeddings
        neighbors: typing.Set[ str ] = set()

        try:
            for entity in entities:
                neighbor_iter = self.strw.w2v_model.wv.most_similar(  # type: ignore
                    positive = [ entity ],
                    topn = num_chunks,
                )

                for neighbor in neighbor_iter:
                    neighbors.add(neighbor)
        except KeyError:
            pass

        if debug:
            ic(neighbors)

        # map the expanded set of entities to nodes in the graph
        expanded_entities: set = entities.union(neighbors)

        anchor_nodes: set = {
            node
            for node, dat in self.strw.sem_overlay.nodes(data = True)
            if "key" in dat and dat["key"] in expanded_entities
        }

        if debug:
            ic(anchor_nodes)

        # extract a subgraph based on shortest paths between anchor nodes
        node_iter: typing.Iterator[ int ] = self.extract_subgraph(
            anchor_nodes,
            debug = debug,
        )

        # extract the chunk neighbors
        chunk_iter: typing.Iterator[ int ] = self.extract_chunk_neighbors(
            list(node_iter),
            debug = debug,
        )

        chunk_ids: typing.List[ int ] = []

        for chunk_id in chunk_iter:
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)

        # enumerate chunks from a vector search -- the basic RAG process
        df_ann_chunk: pl.DataFrame = self.strw.chunk_table.search(  # type: ignore
            question
        ).limit(
            num_chunks
        ).to_polars()

        for row in df_ann_chunk.iter_rows(named = True):
            chunk_id: int =  row["uid"]  # type: ignore

            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)

        if debug:
            ic(chunk_ids)

        # get the text for the combined/ranked list of chunks
        id_list: str = ", ".join([ str(c_id) for c_id in chunk_ids ])
        filter_term: str = f"uid IN ({id_list})"

        chunks: typing.List[ str ] = self.strw.chunk_table.search().where(  # type: ignore
            filter_term
        ).select(
            [ "text" ]
        ).to_polars()["text"].to_list()

        return chunks[:num_chunks]


    def gen_subgraph_paths (
        self,
        anchor_nodes: set,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ str ]:
        """
Generate pairwise shortest paths among the nodes from semantic
expansion, to define a subgraph.

In other words, this emulates a _semantic random walk_.
        """
        for pair in itertools.combinations(anchor_nodes, 2):
            if debug:
                ic(pair)

            for path in nx.all_shortest_paths(self.strw.sem_overlay, pair[0], pair[1]):
                if debug:
                    ic(path)

                for node in path:
                    if node not in pair:
                        dat: dict = self.strw.sem_overlay.nodes[node]

                        if debug:
                            ic(node, dat)

                        yield node


    def extract_subgraph (
        self,
        anchor_nodes: set,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ int ]:
        """
Extract a subgraph, run a _centrality_ algorithm to rerank the most
referenced entities in the subgraph.
        """
        subgraph_iter: typing.Iterator[ str ] = self.gen_subgraph_paths(
            anchor_nodes,
            debug = debug,
        )

        subgraph: nx.Graph = self.strw.sem_overlay.subgraph(
            anchor_nodes.union(set(subgraph_iter))
        )

        rank_iter: dict = nx.pagerank(  # type: ignore
            subgraph,
            self.strw.config["tr"]["tr_alpha"],
        ).items()

        for node, rank in sorted(rank_iter, key = lambda x: x[1], reverse = True):
            dat: dict = self.strw.sem_overlay.nodes[node]

            if debug:
                ic(node, rank, dat)

            yield node


    def extract_chunk_neighbors (
        self,
        ranked_nodes: list,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ int ]:
        """
Find the neighboring chunks for each _anchor node_ in the given list.
        """
        for node in ranked_nodes:
            if debug:
                ic(node)

            for neighbor in self.strw.sem_overlay.neighbors(node):
                dat: dict = self.strw.sem_overlay.nodes[neighbor]

                if dat["kind"] == "Chunk":
                    if debug:
                        ic(neighbor, dat)

                    chunk_id: int = int(neighbor.replace("chunk_", ""))
                    yield chunk_id


    def qa_cycle (
        self,
        question: str,
        *,
        debug: bool = False,
        ) -> baml_types.Response:
        """
Loop to answer questions.
        """
        chunks: typing.List[ str ] = self.get_chunks(
            question,
            debug = debug,
        )

        context: str = "\n".join( chunks )

        response: baml_types.Response = b.RAG(
            question,
            context,
        )

        return response
