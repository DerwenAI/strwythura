#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds assets for constructing a KG, then running GraphRAG downstream.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib
import tomllib
import traceback
import typing
import warnings

from icecream import ic
import gensim
import lancedb
import networkx as nx
import spacy

from .kg import construct_kg
from .rag import GraphRAG
from .valid import TextChunk
from .vis import gen_pyvis


HTML_PATH: str = "kg.html"
KG_PATH: str = "data/kg.json"
W2V_PATH: str = "data/entity.w2v"


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
        self.config: dict = {}

        with open(config_path, mode = "rb") as fp:
            self.config = tomllib.load(fp)

        self.url_list: typing.List[ str ] = []
        self.simple_pipe: spacy.Language = spacy.load(self.config["nlp"]["spacy_model"])
        self.chunk_table: typing.Optional[ lancedb.table.LanceTable ] = None
        self.sem_overlay: nx.Graph = nx.Graph()
        self.w2v_vectors: list = []
        self.w2v_model: typing.Optional[ gensim.models.Word2Vec ] = None


    def build_assets (
        self,
        url_list: typing.List[ str ],
        *,
        debug: bool = False,
        kg_path: str = KG_PATH,
        w2v_path: str = W2V_PATH,
        ) -> int:
        """
Builds assets for constructing a KG.
        """
        self.url_list = url_list

        try:
            # initialize the chunk table
            vect_db: lancedb.db.LanceDBConnection = lancedb.connect(self.config["vect"]["lancedb_uri"])

            self.chunk_table = vect_db.create_table(
                self.config["vect"]["chunk_table"],
                schema = TextChunk,
                mode = "overwrite",
            )

            # construct the graph
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                construct_kg(
                    self.config,
                    self.url_list,
                    self.simple_pipe,
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
        w2v_path: str = W2V_PATH,   
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

        # temporary pathing
        w2v_file: pathlib.Path = pathlib.Path(w2v_path)
        self.w2v_model.save(str(w2v_file))


    def save_graph (
        self,
        *,
        kg_path: str = KG_PATH,
        ) -> None:
        """
Serialize the KG
        """
        with pathlib.Path(KG_PATH).open("w", encoding = "utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(self.sem_overlay, edges = "links"),
                    indent = 2,
                    sort_keys = True,
                )
            )


    def gen_visualization (
        self,
        *,
        html_path: str = HTML_PATH,
        ) -> None:
        """
Generate HTML for an interactive visualization of the graph, based on `PyVis`
        """
        gen_pyvis(
            self.sem_overlay,
            html_path,
            num_docs = len(self.url_list),
        )


    def test_chunks (
        self,
        *,
        debug: bool = False,
        query: str = "how is bacon involved?",
        ) -> None:
        """
Prepare the chunk list from an example query.
        """
        entity: str = " ".join([
            f"{token.pos_}.{token.lemma_}"
            for token in self.simple_pipe(query)
        ])

        rag: GraphRAG = GraphRAG(
            self.chunk_table,
            self.w2v_model,
            self.sem_overlay,
        )

        rag.get_chunks(
            query,
            [ entity ],
            debug = debug,
        )
