#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds assets for constructing a KG, then running GraphRAG downstream.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import itertools
import os
import time
import traceback
import typing

from datasketch import MinHashLSHForest, MinHash
from icecream import ic
from opik.integrations.dspy.callback import OpikCallback
import dspy
import networkx as nx
import opik
import polars as pl
import spacy

from .elem import Entity, EntitySource, NodeKind, STRW_PREFIX
from .work import Workflow


class DSPy_RAG (dspy.Module):
    """
DSPy implementation of a RAG signature.
    """

    def __init__(
        self,
        config: dict,
        project_name: str,
        *,
        run_local: bool = True,
        use_opik: bool = True,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config

        # load the LLM
        if run_local:
            self.lm: dspy.LM = dspy.LM(
                self.config["rag"]["lm_name"],
                api_base = self.config["rag"]["api_base"],
                api_key = "",
                temperature = self.config["rag"]["temperature"],
                max_tokens = self.config["rag"]["max_tokens"],
                stop = None,
                cache = False,
            )
        else:
            OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")

            if OPENAI_API_KEY is None:
                raise ValueError(
                    "Environment variable 'OPENAI_API_KEY' is not set. Please set it to proceed."
                )

            self.lm = dspy.LM(
                "openai/gpt-4o-mini",
                temperature = 0.0,
            )

        # set up `Opik` for observability
        # see: https://www.comet.com/docs/opik/python-sdk-reference/configure.html
        callbacks: list = []

        if use_opik:
            os.environ["OPIK_BASE_URL"] = self.config["opik"]["base_url"]

            opik.configure(
                use_local = True,
                url = self.config["opik"]["base_url"],
            )

            self.opik_callback: OpikCallback = OpikCallback(
                project_name = project_name,
                log_graph = True,
            )

            callbacks.append(self.opik_callback)

        # set up the `DSPy` signature for RAG
        dspy.configure(
            lm = self.lm,
            callbacks = callbacks,
        )

        # define the basic RAG signature
        self.respond: dspy.Predict = dspy.Predict(
            "context, question -> response"
        )

        self.context: list[ str ] = []


    def forward (
        self,
        question: str,
        ) -> dspy.primitives.prediction.Prediction:
        """
Invoke the RAG signature.
        """
        reply: dspy.primitives.prediction.Prediction = self.respond(
            context = self.context,
            question = question,
        )

        return reply


class GraphRAG:
    """
Run a question through `LanceDB` to identify related _chunks_ and
through the `Word2Vec` entity embedding model for _semantic expansion_
to produce a set of _anchor nodes_ in the `NetworkX` ERKG graph.
    """

    def __init__ (
        self,
        work: Workflow,
        project_name: str,
        *,
        run_local: bool = True,
        use_opik: bool = True,
        ) -> None:
        """
Constructor.
        """
        self.work: Workflow = work

        self.rag: DSPy_RAG = DSPy_RAG(
            self.work.config,
            project_name,
            run_local = run_local,
            use_opik = use_opik
        )


    ######################################################################
    ## get anchor nodes and text chunks by leveraging semantics in the ERKG

    def find_entities (
        self,
        question: str,
        *,
        num_perm: int = 128,
        ) -> tuple[set[ int ], list[ MinHash ]]:
        """
Search the entity store for direct matches from NER
        """
        anchor_nodes: set[ int ] = set()
        lem_seq: list[ str ] = []
        doc: spacy.tokens.doc.Doc = self.work.parser.ner_pipe(question)

        lemma_todo: set[ str ] = {
            self.work.parser.tokenize_lemma(span)
            for span in doc.ents
        }

        for sent in doc.sents:
            for item in self.work.parser.transform_sentence(sent):
                lemma_key: str = self.work.parser.tokenize_lemma(item.span)
                lem_seq.append(lemma_key)

                # try to find known entities directly
                if item.label in [ "NOUN" ] and lemma_key not in self.work.parser.STOP_WORDS:
                    found_ent: Entity = self.work.ctx.ent_store.encode_entity(
                        Entity(span = item, lemma_key = lemma_key)
                    )

                    if found_ent is not None:
                        anchor_nodes.add(found_ent.uid)

                # as a fallback, keep the lemma keys for each noun phrase to use in an LSH
                if item.source in [ EntitySource.NER ]:
                    lemma_todo.add(lemma_key)

        # prepare for min hash approximation searches on lemmas
        ner_mh: list[ MinHash ] = []

        # the first hash is special: built on a sequence of lemma keys --
        # one per parsed token -- for the entire sentence
        ner_mh.append(MinHash(num_perm = num_perm))

        for lemma_key in lem_seq:
            for lemma in lemma_key.split(" "):
                ner_mh[0].update(lemma.encode("utf-8"))

        for lemma_key in lemma_todo:
            mh: MinHash = MinHash(num_perm = num_perm)

            for lemma in lemma_key.split(" "):
                mh.update(lemma.encode("utf-8"))

            ner_mh.append(mh)

        return anchor_nodes, ner_mh


    def find_chunks (
        self,
        question: str,
        *,
        max_chunks: int = 11,
        num_perm: int = 128,
        ) -> tuple[ dict[ int, float ], MinHashLSHForest ]:
        """
Search the vector store for chunks in the neighborhood of the question.
(basic RAG process)
        """
        rag_chunks: dict[ int, float ] = {}
        chunk_nodes: dict[ int, float ] = {}
        sem_rel: str = f"{STRW_PREFIX}within_chunk"

        chunk_list: list[ dict ] = self.work.ctx.chunk_table.search(
            question
        ).select(
            [ "uid", "_distance" ]
        ).limit(
            max_chunks
        ).to_list()

        for row in chunk_list:
            chunk_id: int = row["uid"]
            distance: float = round((100.0 - row["_distance"]) / 100.0, 4)
            rag_chunks[chunk_id] = distance

            for node_id, _, keys, weight in self.work.ctx.erkg.in_edges(
                nbunch = f"chunk_{chunk_id}",
                    data = "weight",
                    keys = True,
            ):
                if sem_rel in keys:
                    metric: float = round(weight * distance, 4)

                    if node_id not in chunk_nodes:
                        chunk_nodes[node_id] = metric
                    else:
                        chunk_nodes[node_id] = max(chunk_nodes[node_id], metric)

        # create a MinHash LSH Forest
        forest: MinHashLSHForest = MinHashLSHForest(num_perm = num_perm)

        for node_id, metric in chunk_nodes.items():
            hit: dict = self.work.ctx.erkg.nodes[node_id]
            mh_hit: MinHash = MinHash(num_perm = num_perm)

            for lemma in hit["lemma"].split(" "):
                mh_hit.update(lemma.encode("utf-8"))

            forest.add(node_id, mh_hit)

        forest.index()

        return rag_chunks, forest


    def get_anchor_nodes (
        self,
        question: str,
        *,
        max_chunks: int = 11,
        num_perm: int = 128,
        lsh_top_k_question: int = 9,
        lsh_top_k_lemma: int = 3,
        ) -> tuple[ set[ int ], dict[ int, float ]]:
        """
Find the anchor nodes to use for enhanced GraphRAG.
        """
        # find entities in the neighborhood of the question
        anchor_nodes: set[ int ] = set()
        ner_mh: list[ MinHash ] = []

        anchor_nodes, ner_mh = self.find_entities(
            question,
            num_perm = num_perm,
        )

        # find chunks in the neighborhood of the question
        rag_chunks: dict[ int, float ] = {}
        forest: MinHashLSHForest | None = None

        rag_chunks, forest = self.find_chunks(
            question,
            max_chunks = max_chunks,
            num_perm = num_perm,
        )

        for node_id in forest.query(ner_mh[0], lsh_top_k_question):
            anchor_nodes.add(node_id)

        for mh_item in ner_mh[1:]:
            for node_id in forest.query(mh_item, lsh_top_k_lemma):
                anchor_nodes.add(node_id)

        return anchor_nodes, rag_chunks


    ######################################################################
    ## semantic expansion and random walks

    def extract_question_subgraph (
        self,
        anchor_nodes: set[ int ],
        *,
        debug: bool = False,
        ) -> typing.Iterator[ int ]:
        """
Extract a subgraph, then run a _centrality_ algorithm to rerank the
most-referenced entities in the subgraph.
        """
        subgraph_iter: typing.Iterator[ str ] = self.semantic_random_walk(
            anchor_nodes,
        )

        subgraph: nx.MultiDiGraph = self.work.ctx.erkg.subgraph(
            anchor_nodes.union(set(subgraph_iter))
        )

        rank_iter: dict = nx.pagerank(
            subgraph,
            self.work.config["tr"]["tr_alpha"],
        ).items()

        for node, rank in sorted(rank_iter, key = lambda x: x[1], reverse = True):
            dat: dict = self.work.ctx.erkg.nodes[node]

            if debug:
                ic(node, rank, dat)

            yield node


    def semantic_random_walk (
        self,
        anchor_nodes: set[ int ],
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

            try:
                for path in nx.all_shortest_paths(self.work.ctx.erkg, pair[0], pair[1]):
                    if debug:
                        ic(path)

                    for node in path:
                        if node not in pair:
                            dat: dict = self.work.ctx.erkg.nodes[node]

                            if debug:
                                ic(node, dat)

                            yield node
            except nx.NetworkXNoPath:
                # ignore attempts when the source node is unreachable
                pass


    def find_chunk_neighbors (
        self,
        anchor_nodes: set[ int ],
        *,
        debug: bool = False,
        ) -> typing.Iterator[ int ]:
        """
Find the neighboring chunks for each _anchor node_.
        """
        for node in anchor_nodes:
            if debug:
                ic(node)

            for neighbor in self.work.ctx.erkg.neighbors(node):
                dat: dict = self.work.ctx.erkg.nodes[neighbor]

                if dat["kind"] == NodeKind.CHUNK.value:
                    if debug:
                        ic(neighbor, dat)

                    chunk_id: int = int(neighbor.replace("chunk_", ""))
                    yield chunk_id


    def perform_semantic_expansion (
        self,
        anchor_nodes: set[ int ],
        *,
        w2v_top_k: int = 20,
        w2v_min_dist: float = 0.33,
        ) -> None:
        """
Perform a semantic expansion using entity embeddings.
        """
        neighbors: dict[ int, float ] = {}

        for node_id in anchor_nodes:
            anchor_node: dict = self.work.ctx.erkg.nodes[node_id]

            try:
                for uid, distance in self.work.ctx.ent_store.w2v_model.wv.most_similar(
                    str(node_id),
                    topn = w2v_top_k,
                ):
                    if distance <= w2v_min_dist:
                        neigh_id: int = int(uid)
                        neighbors[neigh_id] = round(distance, 4)
            except KeyError:
                pass

        for neigh_id, distance in sorted(neighbors.items(), key = lambda x: x[1]):
            neighbor: dict = self.work.ctx.erkg.nodes[neigh_id]

            if "source" in neighbor and EntitySource(neighbor["source"]) <= EntitySource.NER:
                anchor_nodes.add(neigh_id)


    def get_chunks_text (
        self,
        rag_chunks: dict[ int, float ],
        ) -> list[ str ]:
        """
Retrieve text for the combined list of chunks.
        """
        id_list: str = ", ".join([ str(c_id) for c_id in rag_chunks.keys() ])

        chunks: list[ str ] = self.work.ctx.chunk_table.search().where(
            f"uid IN ({id_list})"
        ).select(
            [ "text" ]
        ).to_polars()["text"].to_list()

        return chunks


    def run_errag (
        self,
        question: str,
        *,
        debug: bool = False,
        ) -> list[ str ]:
        """
Run an enchanced GraphRAG to prioritize and retrieve text chunks by
leveraging the ERKG and entity embeddings.
        """
        anchor_nodes: set[ int ] = set()
        rag_chunks: dict[ int, float ] = {}

        anchor_nodes, rag_chunks = self.get_anchor_nodes(
            question,
            max_chunks = self.work.config["rag"]["max_chunks"],
            num_perm = self.work.config["rag"]["num_perm"],
            lsh_top_k_question = self.work.config["rag"]["lsh_top_k_question"],
            lsh_top_k_lemma = self.work.config["rag"]["lsh_top_k_lemma"],
        )

        if debug:
            ic(rag_chunks)

            for node_id in anchor_nodes:
                anchor_node: dict = self.work.ctx.erkg.nodes[node_id]
                ic(anchor_node)

        # perform a semantic expansion using entity embeddings
        self.perform_semantic_expansion(
            anchor_nodes,
            w2v_top_k = self.work.config["rag"]["w2v_top_k"],
            w2v_min_dist = self.work.config["rag"]["w2v_min_dist"],
        )

        # extract a subgraph constructed from the shortest paths
        # between anchor nodes
        subgraph: set[ int ] = set(self.extract_question_subgraph(anchor_nodes))

        if debug:
            ic(subgraph)

        # add the chunks for each anchor node
        for chunk_id in self.find_chunk_neighbors(subgraph):
            if chunk_id not in rag_chunks:
                rag_chunks[chunk_id] = 0.5 # impute to median distance 

        if debug:
            ic(rag_chunks)

        # retrieve text for the combined list of chunks
        chunks: list[ str ] = self.get_chunks_text(rag_chunks)

        return chunks


    ######################################################################
    ## question/answer

    def qa_signature (
        self,
        question: str,
        chunks: list[ str ],
        ) -> dspy.primitives.prediction.Prediction:
        """
Run one question/answer cycle.
        """
        self.rag.context = chunks
        response: dspy.primitives.prediction.Prediction = self.rag(question)

        if False: # disable for now; too verbose
            dspy.inspect_history()

        return response


    def question_answer (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """
Loop to answer questions.
        """
        try:
            # loop to answer questions
            while True:
                question: str = input("\nQuoi? ").strip()

                if len(question) < 1:
                    continue

                if question.lower() in [ "quitter", "bye", "adieu" ]:
                    break

                # enchanced GraphRAG prioritizes and retrieves text chunks
                chunks: list[ str ] = self.run_errag(
                    question,
                    debug = debug,
                )

                # LLM summarizes the text chunks in response to the question
                response: dspy.primitives.prediction.Prediction = self.qa_signature(
                    question,
                    chunks,
                )

                ic(question)
                ic(response.response)
                print("-" * 10)

        except EOFError:
            print("")
        except Exception as ex:
            ic(ex)
            traceback.print_exc()
        finally:
            print("\nÀ bientôt!\n")
            time.sleep(.1)
