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

from .ctx import TextChunk
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

        # search assets
        self.anchor_nodes: set[ str ] = set()
        self.rag_chunks: dict[ str, float ] = {}


    ######################################################################
    ## question/answer

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


    ######################################################################
    ## enhanced GraphRAG methods

    def run_errag (
        self,
        question: str,
        *,
        disable_graph: bool = False,
        debug: bool = True, # False
        ) -> list[ str ]:
        """
Run an enchanced GraphRAG to retrieve and prioritize text chunks
by leveraging the ERKG and entity embeddings.
        """
        max_chunks: int = self.work.config["rag"]["max_chunks"]
        num_perm: int = self.work.config["rag"]["num_perm"]

        # find the text chunks which are nearest to the question,
        # then find the entity nodes linked to these chunks
        chunk_nodes: dict[ str, float ] = self.find_rag_chunks(
            question,
            max_chunks = max_chunks,
            num_perm = num_perm,
        )

        if disable_graph:
            # disable the GraphRAG aspects, using RAG-only --
            # for testing and evaluation purposes
            return

        # find entities in the neighborhood of the question,
        # identifying the initial set of anchor nodes, plus MinHash
        # digests for the lemmatized phrases among these entities
        ner_mh: list[ MinHash ] = self.find_nearby_entities(
            question,
            num_perm = num_perm,
        )

        # use a locality-sensitive hash to filter the entity nodes
        # linked to chunks, to augment the set of anchor nodes
        self.augment_anchor_nodes(
            chunk_nodes,
            ner_mh,
            num_perm = num_perm,
        )

        # purely for debugging
        if debug:
            ic(self.rag_chunks)

            for node_id in self.anchor_nodes:
                anchor_node: dict = self.work.ctx.erkg.nodes[node_id]
                ic(anchor_node)

        # perform a semantic expansion using entity embeddings
        self.perform_semantic_expansion(
            w2v_top_k = self.work.config["rag"]["w2v_top_k"],
            w2v_min_dist = self.work.config["rag"]["w2v_min_dist"],
        )

        # extract a subgraph constructed from the shortest paths
        # between anchor nodes
        subgraph: set[ str ] = set(list(self.extract_question_subgraph()))

        if debug:
            ic(subgraph)


    def find_rag_chunks (
        self,
        question: str,
        *,
        max_chunks: int = 11,
        num_perm: int = 128,
        debug: bool = False,
        ) -> dict[ str, float ]:
        """
Search the vector store for text chunks in the neighborhood of the
`question` prompt, which is the basic RAG process.

Then find graph nodes for entities linked to the selected chunks,
which are returned as a dictionary.
        """
        chunk_list: list[ dict ] = self.work.ctx.chunk_table.search(
            question
        ).select(
            [ "uid", "_distance" ]
        ).limit(
            max_chunks
        ).to_list()

        chunk_nodes: dict[ str, float ] = {}
        sem_rel: str = f"{STRW_PREFIX}within_chunk"

        for row in chunk_list:
            chunk_id: int = row["uid"]
            chunk_iri: str = TextChunk.get_iri(chunk_id)
            distance: float = round((100.0 - row["_distance"]) / 100.0, 4)

            if debug:
                ic(chunk_iri, distance)

            self.rag_chunks[chunk_id] = distance

            for node_id, _, keys, weight in self.work.ctx.erkg.in_edges(
                nbunch = chunk_iri,
                data = "weight",
                keys = True,
            ):
                if sem_rel in keys:
                    metric: float = round(weight * distance, 4)

                    if node_id not in chunk_nodes:
                        chunk_nodes[node_id] = metric
                    else:
                        chunk_nodes[node_id] = max(chunk_nodes[node_id], metric)

        return chunk_nodes


    def find_nearby_entities (
        self,
        question: str,
        *,
        num_perm: int = 128,
        debug: bool = False,
        ) -> list[ MinHash ]:
        """
Search the entity store for direct matches from NER
        """
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
                        self.anchor_nodes.add(found_ent.get_iri())

                # as a fallback, keep the lemma keys for each noun phrase to use in an LSH
                if item.source in [ EntitySource.NER ]:
                    lemma_todo.add(lemma_key)

        if debug:
            ic(lemma_todo)
            ic(lem_seq)
            ic(self.anchor_nodes)

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

        return ner_mh


    def augment_anchor_nodes (
        self,
        chunk_nodes: dict[ str, float ],
        ner_mh: list[ MinHash ],
        *,
        num_perm: int = 128,
        debug: bool = False,
        ) -> None:
        """
Use a _locality-sensitive hash_ (LSH) to filter the entity nodes
linked to chunks, to augment the set of anchor nodes.
        """
        lsh_top_k_question: int = self.work.config["rag"]["lsh_top_k_question"]
        lsh_top_k_lemma: int = self.work.config["rag"]["lsh_top_k_lemma"]

        # index a MinHash LSH Forest of lemmatized terms among
        # the entity nodes linked to chunks
        forest: MinHashLSHForest = MinHashLSHForest(
            num_perm = num_perm,
        )

        for node_id, metric in chunk_nodes.items():
            hit: dict = self.work.ctx.erkg.nodes[node_id]

            if "lemma" in hit:
                mh_hit: MinHash = MinHash(num_perm = num_perm)

                for lemma in hit["lemma"].split(" "):
                    mh_hit.update(lemma.encode("utf-8"))

                forest.add(node_id, mh_hit)

        forest.index()

        # filter the entity nodes linked to chunks, retaining those
        # closest to the `question`, to augment the anchor nodes
        for node_id in forest.query(ner_mh[0], lsh_top_k_question):
            self.anchor_nodes.add(node_id)

        for mh_item in ner_mh[1:]:
            for node_id in forest.query(mh_item, lsh_top_k_lemma):
                self.anchor_nodes.add(node_id)


    def perform_semantic_expansion (
        self,
        *,
        w2v_top_k: int = 20,
        w2v_min_dist: float = 0.33,
        debug: bool = True,
        ) -> None:
        """
Perform a _semantic expansion_ using entity embeddings, with the set of
anchor nodes as the starting points.
        """
        decoder: dict[ int, Entity ] = self.work.ctx.ent_store.get_decoder()
        neighbors: dict[ str, float ] = {}

        for node_id in self.anchor_nodes:
            try:
                anchor_node: dict = self.work.ctx.erkg.nodes[node_id]

                if "lemma" in anchor_node:
                    lemma_key: str = anchor_node["lemma"]
                    ent: Entity = self.work.ctx.ent_store.entities[lemma_key]

                    if debug:
                        ic(node_id, lemma_key)

                    for uid, distance in self.work.ctx.ent_store.w2v_model.wv.most_similar(
                        str(ent.uid),
                        topn = w2v_top_k,
                    ):
                        if distance <= w2v_min_dist:
                            neigh_iri: str = decoder[int(uid)].get_iri()

                            if debug:
                                ic(neigh_iri, distance)

                            neighbors[neigh_iri] = round(distance, 4)

            except KeyError as ex:
                ic(node_id)
                ic(ex)
                traceback.print_exc()

        for neigh_iri, distance in sorted(neighbors.items(), key = lambda x: x[1]):
            neighbor: dict = self.work.ctx.erkg.nodes[neigh_iri]

            if "method" in neighbor and EntitySource(neighbor["method"]) <= EntitySource.NER:
                ic("ADD", neigh_iri)

                self.anchor_nodes.add(neigh_iri)


    def extract_question_subgraph (
        self,
        *,
        debug: bool = True, # False
        ) -> typing.Iterator[ str ]:
        """
Extract a subgraph, then run a _centrality_ algorithm to rerank the
most-referenced entities in the subgraph.
        """
        walks: set[ str ] = set(list(self.semantic_random_walk()))

        if debug:
            ic(walks)

        subgraph: nx.MultiDiGraph = self.work.ctx.erkg.subgraph(
            self.anchor_nodes.union(walks)
        )

        rank_iter: dict = nx.pagerank(
            subgraph,
            self.work.config["tr"]["tr_alpha"],
        ).items()

        for node_id, rank in sorted(rank_iter, key = lambda x: x[1], reverse = True):
            if debug:
                hit: dict = self.work.ctx.erkg.nodes[node_id]
                ic("tr", node_id, rank, hit)

            yield node_id


    def semantic_random_walk (
        self,
        *,
        debug: bool = True, # False
        ) -> typing.Iterator[ str ]:
        """
Generate pairwise shortest paths among the nodes from semantic
expansion, to define a subgraph.

In other words, this emulates a _semantic random walk_.
        """
        for pair in itertools.combinations(self.anchor_nodes, 2):
            if debug:
                ic(pair)

            try:
                for path in nx.all_shortest_paths(self.work.ctx.erkg, pair[0], pair[1]):
                    if debug:
                        ic(path)

                    for node_id in path:
                        if node_id not in pair:
                            if debug:
                                hit: dict = self.work.ctx.erkg.nodes[node_id]
                                ic("walk", node_id, hit)

                            yield node_id
            except nx.NetworkXNoPath:
                # ignore attempts when the source node is unreachable
                pass
