#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds assets for constructing a KG, then running GraphRAG downstream.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import itertools
import logging
import os
import pathlib
import tomllib
import traceback
import typing
import warnings

from icecream import ic  # type: ignore
import networkx as nx
import polars as pl
import spacy
import transformers  # type: ignore

from .baml_client import b
from .baml_client import types as baml_types
from .context import DomainContext
from .elem import NodeKind
from .kg import KnowledgeGraph
from .nlp import Parser
from .vis import VisHTML


class Strwythura:  # pylint: disable=R0902
    """
Builds assets for constructing a KG, then running GraphRAG downstream.
    """

    def __init__ (
        self,
        domain_context: DomainContext,
        *,
        config_path: pathlib.Path = pathlib.Path("config.toml"),
        overwrite: bool = False,
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

        # initialize the data structures used for assets
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.domain_context: DomainContext = domain_context
            self.domain_context.set_config(self.config)

            if overwrite:
                self.domain_context.init_chunk_table()
                self.domain_context.load_taxonomy()

            self.parser: Parser = Parser(self.config)
            self.simple_pipe: spacy.Language = spacy.load(self.config["nlp"]["spacy_model"])
            self.entity_pipe: spacy.Language = self.parser.build_entity_pipe(
                self.domain_context.get_ner_labels()
            )


    def build_assets (
        self,
        url_list: typing.List[ str ],
        *,
        debug: bool = False,
        ) -> None:
        """
Builds assets for constructing a KG.
        """
        try:
            kg: KnowledgeGraph = KnowledgeGraph(self.config)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                kg.build_graph(
                    url_list,
                    self.domain_context,
                    self.parser,
                    self.simple_pipe,
                    self.entity_pipe,
                    debug = debug,
                )

        except Exception as ex:  # pylint: disable=W0718
            ic(ex)
            traceback.print_exc()


    def gen_visualization (
        self,
        url_list: typing.List[ str ],
        vis_html: VisHTML,
        *,
        html_path: typing.Optional[ pathlib.Path ] = None,
        ) -> None:
        """
Generate HTML for an interactive visualization of the graph.
        """
        if html_path is None:
            html_path = pathlib.Path(self.config["kg"]["html_path"])

        vis_html.set_config(self.config)

        vis_html.gen_vis_html(
            html_path.as_posix(),
            self.domain_context.vis_nodes(len(url_list)),
            self.domain_context.vis_edges(),
        )


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
        *,
        debug: bool = False,
        ) -> typing.Iterator[ str ]:
        """
Extract entity spans from a text question, generating their lemma keys.
        """
        doc: spacy.tokens.doc.Doc = self.strw.entity_pipe(question)  # type: ignore  # pylint: disable=I1101

        for span in doc.ents:
            lemma_key: str = self.strw.domain_context.parse_lemma(span)  # type: ignore

            if debug:
                ic(span, lemma_key)

            yield lemma_key


    def get_chunks (  # pylint: disable=R0912,R0914
        self,
        question: str,
        *,
        debug: bool = False,
        num_chunks: typing.Optional[ int ] = None,
        ) -> typing.List[ str ]:
        """
Run semantic search to produce a set of text chunks.
        """
        chunk_ids: typing.List[ int ] = []

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
                neighbor_iter = self.strw.domain_context.w2v_model.wv.most_similar(  # type: ignore
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
            for node, dat in self.strw.domain_context.sem_layer.nodes(data = True)
            if "key" in dat and dat["key"] in expanded_entities
        }

        if debug:
            ic(anchor_nodes)

        # trap edge cases where there is no connected subgraph
        try:
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

            for chunk_id in chunk_iter:
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
        except nx.exception.NetworkXNoPath:
            pass

        # enumerate chunks from a vector search -- the basic RAG process
        df_ann_chunk: pl.DataFrame = self.strw.domain_context.chunk_table.search(  # type: ignore
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

        chunks: typing.List[ str ] = self.strw.domain_context.chunk_table.search().where(  # type: ignore  # pylint: disable=C0301
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

            for path in nx.all_shortest_paths(self.strw.domain_context.sem_layer, pair[0], pair[1]):
                if debug:
                    ic(path)

                for node in path:
                    if node not in pair:
                        dat: dict = self.strw.domain_context.sem_layer.nodes[node]

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

        subgraph: nx.MultiDiGraph = self.strw.domain_context.sem_layer.subgraph(  # type: ignore
            anchor_nodes.union(set(subgraph_iter))
        )

        rank_iter: dict = nx.pagerank(  # type: ignore
            subgraph,
            self.strw.config["tr"]["tr_alpha"],
        ).items()

        for node, rank in sorted(rank_iter, key = lambda x: x[1], reverse = True):
            dat: dict = self.strw.domain_context.sem_layer.nodes[node]

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

            for neighbor in self.strw.domain_context.sem_layer.neighbors(node):
                dat: dict = self.strw.domain_context.sem_layer.nodes[neighbor]

                if dat["kind"] == NodeKind.CHUNK.value:
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
