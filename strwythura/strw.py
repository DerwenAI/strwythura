#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds assets for constructing a KG, then running GraphRAG downstream.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import logging
import os
import pathlib
import tomllib
import traceback
import typing
import warnings

from icecream import ic
import gensim
import lancedb
import networkx as nx
import pandas as pd
import spacy
import transformers

from .baml_client import b
from .baml_client import types as baml_types
from .graph import TextChunk
from .kg import KnowledgeGraph
from .nlp import Parser
from .vis import gen_pyvis


class Strwythura:
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
        #loggers: dict = { name:logging.getLogger(name) for name in logging.root.manager.loggerDict }
        #ic(loggers)
        #logging.getLogger("glirel.spacy_integration").setLevel(logging.ERROR)

        # initial data structures for assets
        self.parser: Parser = Parser(self.config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.simple_pipe: spacy.Language = spacy.load(self.config["nlp"]["spacy_model"])
            self.entity_pipe: spacy.Language = self.parser.build_entity_pipe()
            self.chunk_table: typing.Optional[ lancedb.table.LanceTable ] = None
            self.sem_overlay: nx.Graph = nx.Graph()
            self.w2v_vectors: list = []
            self.w2v_model: typing.Optional[ gensim.models.Word2Vec ] = None


    def build_assets (
        self,
        url_list: typing.List[ str ],
        ner_labels: typing.List[ str ],
        *,
        debug: bool = False,
        kg_path: typing.Optional[ pathlib.Path ] = None,
        w2v_path: typing.Optional[ pathlib.Path ] = None,
        ) -> int:
        """
Builds assets for constructing a KG.
        """
        self.parser.update_data(
            url_list,
            ner_labels,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                # initialize the chunk table
                vect_db: lancedb.db.LanceDBConnection = lancedb.connect(self.config["vect"]["lancedb_uri"])

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

            except Exception as ex:
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
        w2v_max: int = max([
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
                        edges = "links",
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
        ) -> int:
        """
Load the serialized assets for a constructed KG.
        """
        if w2v_path is None:
            w2v_path = pathlib.Path(self.config["ent"]["w2v_path"])

        self.w2v_model = gensim.models.Word2Vec.load(w2v_path.as_posix())

        if kg_path is None:
            kg_path = pathlib.Path(self.config["kg"]["kg_path"])

        with pathlib.Path(kg_path).open("r", encoding = "utf-8") as fp:
            self.sem_overlay = nx.node_link_graph(
                json.load(fp),
                edges = "links",
            )

        vect_db: lancedb.db.LanceDBConnection = lancedb.connect(self.config["vect"]["lancedb_uri"])
        self.chunk_table = vect_db.open_table(self.config["vect"]["chunk_table"])


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
        doc: spacy.tokens.doc.Doc = list(
            self.strw.entity_pipe.pipe(
                [( question, Parser.RE_LABELS )],
                as_tuples = True,
            )
        )[0][0]

        for span in doc.ents:
            key: str = " ".join([
                tok.pos_ + "." + tok.lemma_.strip().lower()
                for tok in span
            ])
        
            yield key


    def get_chunks (
        self,
        question: str,
        *,
        debug: bool = False,
        num_chunks: int = 10,
        ) -> typing.List[ str ]:
        """
Run semantic search to produce a set of text chunks.
        """
        # show the question
        if debug:
            ic(question)

        # enumerate chunks from a vector search -- the basic RAG process
        df_chunk: pd.DataFrame = self.strw.chunk_table.search(question).to_pandas()

        if debug:
            ic(df_chunk)

            for row in df_chunk.itertuples():
                ic(row.text)

        # enumerate the nearest neighbor entities from the entity embedding model
        neighbors: list = []

        for entity in self.find_entities(question):
            try:
                neighbor_iter = self.strw.w2v_model.wv.most_similar(
                    positive = [ entity ],
                    topn = num_chunks,
                )

                for neighbor in neighbor_iter:
                    neighbors.append(neighbor)
            except KeyError:
                pass

        df_entity: pd.DataFrame = pd.DataFrame([
            {
                "entity": neighbor[0],
                "distance": neighbor[1],
            }
            for neighbor in neighbors
            if neighbor[1] > 0.0
        ])

        if debug:
            ic(df_entity)

        # perform a semantic expansion to enrich the anchor nodes
        if len(df_entity) > 0:
            expansion: typing.Set[ str ] = set(df_entity["entity"].values.tolist())

            for node, dat in self.strw.sem_overlay.nodes(data = True):
                if "key" in dat and dat["key"] in expansion:
                    if debug:
                        ic(node, dat)

        # return a list of text chunks
        return [
            row.text
            for row in df_chunk.itertuples()
        ]


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
