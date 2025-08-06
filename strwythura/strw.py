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
import pandas as pd
import spacy

from .kg import construct_kg
from .nlp import RE_LABELS, init_nlp_pipe
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.url_list: typing.List[ str ] = []
            self.simple_pipe: spacy.Language = spacy.load(self.config["nlp"]["spacy_model"])
            self.entity_pipe: spacy.Language = init_nlp_pipe(self.config)
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
                construct_kg(
                    self.url_list,
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


    def load_assets (
        self,
        *,
        kg_path: str = KG_PATH,
        w2v_path: str = W2V_PATH,
        ) -> int:
        """
Load the serialized assets for a constructed KG.
        """
        self.w2v_model = gensim.models.Word2Vec.load(w2v_path)

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


    def extract_entities (
        self,
        question: str,
        ) -> typing.Iterator[ str ]:
        """
Extract entity spans from a text question.
        """
        doc: spacy.tokens.doc.Doc = list(
            self.strw.entity_pipe.pipe(
                [( question, RE_LABELS )],
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

        for entity in self.extract_entities(question):
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
