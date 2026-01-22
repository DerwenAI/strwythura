#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run an enhanced GraphRAG based on ERKG and entity embedding, using DSPy.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import itertools
import os
import time
import traceback
import typing
import warnings

from datasketch import MinHashLSHForest, MinHash  # type: ignore
from icecream import ic
from opik.integrations.dspy.callback import OpikCallback
import dspy  # type: ignore
import networkx as nx
import opik
import spacy

from .ctx import TextChunk
from .elem import Entity, EntitySource, NodeKind, STRW_PREFIX
from .work import Workflow


class TracedCallback (OpikCallback):
    """Modify the Opik callback to capture trace info for DSPy integration."""

    def __init__ (
        self,
        project_name: str | None = None,
        log_graph: bool = False,
        ) -> None:
        """
Constructor.
        """
        self.last_trace_id: str | None = None
        super().__init__(project_name, log_graph)


    def _end_trace (
        self,
        call_id: str,
        ) -> None:
        """
Override to capture the `trace_id` for this call to DSPy.
        """
        self.last_trace_id = self._map_call_id_to_trace_data[call_id].id
        super()._end_trace(call_id)


class DSPy_RAG (dspy.Module):  # pylint: disable=C0103
    """
DSPy implementation of a RAG signature.
    """

    def __init__(  # pylint: disable=W0231
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
        self.project_name: str = project_name

        # load the LLM
        dspy.configure(
            track_usage = True,
        )

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
            OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")  # type: ignore  # pylint: disable=C0103

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

            self.opik_callback: TracedCallback = TracedCallback(
                project_name = self.project_name,
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
        project_description: str,
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
            use_opik = use_opik,
        )

        # search assets
        self.description: str = project_description
        self.anchor_nodes: set[ str ] = set()
        self.rag_chunks: dict[ int, float ] = {}
        self.subgraph: nx.Graph = nx.Graph()


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
                self.run_errag(
                    question,
                    debug = debug,
                )

                # LLM summarizes the text chunks in response to the question
                with warnings.catch_warnings(record=True) as caught_warnings:  # pylint: disable=W0612
                    warnings.simplefilter("always")  # catch all warnings

                    response: dspy.primitives.prediction.Prediction = self.qa_signature(
                        question,
                        self.get_chunks_text(),
                    )

                ic(question)
                ic(response.response)
                ic(self.rag)
                print("-" * 10)

        except EOFError:
            print("")
        except Exception as ex:  # pylint: disable=W0718
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

        if False: # disable for now; too verbose  # pylint: disable=W0125
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
        ) -> None:
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

            for iri in self.anchor_nodes:
                anchor: dict = self.work.ctx.erkg.get_node(iri)
                ic(anchor)

        # perform a semantic expansion using entity embeddings
        self.perform_semantic_expansion(
            w2v_top_k = self.work.config["rag"]["w2v_top_k"],
            w2v_min_dist = self.work.config["rag"]["w2v_min_dist"],
        )

        # extract a subgraph constructed from the shortest paths
        # between anchor nodes
        # TODO: if this variable isn't used, let's refactor it out
        try:
            subgraph: set[ str ] = set(list(self.extract_question_subgraph()))  # pylint: disable=W0612
        except Exception as ex:  # pylint: disable=W0718
            ic(ex)

        # add the chunks for each anchor node
        for chunk_id in self.find_chunk_neighbors():
            if chunk_id not in self.rag_chunks:
                # impute to median distance
                self.rag_chunks[chunk_id] = 0.5


    def find_rag_chunks (
        self,
        question: str,
        *,
        max_chunks: int = 11,
        debug: bool = False,
        ) -> dict[ str, float ]:
        """
Search the vector store for text chunks in the neighborhood of the
`question` prompt, which is the basic RAG process.

Then find graph nodes for entities linked to the selected chunks,
which are returned as a dictionary.
        """
        chunk_list: list[ dict ] = self.work.ctx.chunk_table.search(  # type: ignore
            question
        ).select(
            [ "uid", "_distance" ]
        ).limit(
            max_chunks
        ).to_list()

        chunk_nodes: dict[ str, float ] = {}
        sem_rel: str = f"{STRW_PREFIX}within_chunk"

        for row in chunk_list:
            chunk_id: int = int(row["uid"])
            chunk_iri: str = TextChunk.get_iri(chunk_id)
            distance: float = round((100.0 - row["_distance"]) / 100.0, 4)

            if debug:
                ic(chunk_iri, distance)

            self.rag_chunks[chunk_id] = distance

            for ent_iri, keys, val in self.work.ctx.erkg.inbound_edges(chunk_iri, attr = "weight"):  # type: ignore
                if sem_rel in keys:
                    metric: float = round(val * distance, 4)

                    if ent_iri not in chunk_nodes:
                        chunk_nodes[ent_iri] = metric
                    else:
                        chunk_nodes[ent_iri] = max(chunk_nodes[ent_iri], metric)

        return chunk_nodes


    def find_nearby_entities (
        self,
        question: str,
        *,
        num_perm: int = 128,
        debug: bool = False, # True
        ) -> list[ MinHash ]:
        """
Search the entity store for direct matches from NER
        """
        lem_seq: list[ str ] = []
        doc: spacy.tokens.doc.Doc = self.work.parser.ner_pipe(question)  # type: ignore

        lemma_todo: set[ str ] = {
            self.work.parser.tokenize_lemma(span)  # type: ignore
            for span in doc.ents
        }

        for sent in doc.sents:
            for item in self.work.parser.transform_sentence(sent):  # type: ignore
                lemma_key: str = self.work.parser.tokenize_lemma(item.span)  # type: ignore
                lem_seq.append(lemma_key)

                if debug:
                    ic(item, lemma_key)

                # try to find known entities directly
                # NB: this only works for single-word phrases
                if item.label in [ "ADP", "NOUN" ] and lemma_key not in self.work.parser.STOP_WORDS:  # type: ignore
                    found_ent: Entity = self.work.ctx.ent_store.encode_entity(  # type: ignore
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

                if debug:
                    ic("mh0", lemma)

        for lemma_key in lemma_todo:
            mh: MinHash = MinHash(num_perm = num_perm)

            for lemma in lemma_key.split(" "):
                mh.update(lemma.encode("utf-8"))

                if debug:
                    ic("mh", lemma)

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
        # index a MinHash LSH Forest of lemmatized terms
        forest: MinHashLSHForest = MinHashLSHForest(
            num_perm = num_perm,
        )

        # add lemma keys for entity nodes linked to chunks
        for iri, _ in chunk_nodes.items():
            hit: dict = self.work.ctx.erkg.get_node(iri)

            if "lemma" in hit:
                mh_hit: MinHash = MinHash(num_perm = num_perm)

                for lemma in hit["lemma"].split(" "):
                    mh_hit.update(lemma.encode("utf-8"))

                forest.add(iri, mh_hit)

        # add lemma keys for entity resolution results
        for lemma_key, ent in self.work.ctx.ent_store.entities.items():
            if ent.span.source == EntitySource.ER:
                mh_hit = MinHash(num_perm = num_perm)

                for lemma in lemma_key.split(" "):
                    mh_hit.update(lemma.encode("utf-8"))

                ent_iri: str = ent.get_iri()

                if not ent_iri in forest:
                    forest.add(ent_iri, mh_hit)

        forest.index()

        # filter the entity nodes linked to chunks, retaining those
        # closest to the `question`, to augment the anchor nodes
        lsh_top_k_question: int = self.work.config["rag"]["lsh_top_k_question"]
        lsh_top_k_lemma: int = self.work.config["rag"]["lsh_top_k_lemma"]

        for iri in forest.query(ner_mh[0], lsh_top_k_question):
            self.anchor_nodes.add(iri)

        for mh_item in ner_mh[1:]:
            for iri in forest.query(mh_item, lsh_top_k_lemma):
                self.anchor_nodes.add(iri)

        if debug:
            ic(self.anchor_nodes)


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

        for iri in self.anchor_nodes:
            try:
                anchor: dict = self.work.ctx.erkg.get_node(iri)

                if "lemma" in anchor:
                    lemma_key: str = anchor["lemma"]
                    ent: Entity = self.work.ctx.ent_store.entities[lemma_key]

                    if debug:
                        ic(iri, lemma_key)

                    for uid, distance in self.work.ctx.ent_store.w2v_model.wv.most_similar(  # type: ignore
                        str(ent.uid),
                        topn = w2v_top_k,
                    ):
                        if distance <= w2v_min_dist:
                            neigh_iri: str = decoder[int(uid)].get_iri()

                            if debug:
                                ic(neigh_iri, distance)

                            neighbors[neigh_iri] = round(distance, 4)

            except KeyError as ex:
                # TODO: using logging to trace these embedding errors?
                print("w2v_model.wv.most_similar:", iri)
                ic(ex)

        for neigh_iri, distance in sorted(neighbors.items(), key = lambda x: x[1]):
            neigh_hit: dict = self.work.ctx.erkg.get_node(neigh_iri)

            if "method" in neigh_hit and EntitySource(neigh_hit["method"]) <= EntitySource.NER:
                self.anchor_nodes.add(neigh_iri)

                if debug:
                    ic("ADD", neigh_iri, neigh_hit)


    def extract_question_subgraph (
        self,
        *,
        debug: bool = False, # True
        ) -> typing.Iterator[ str ]:
        """
Extract a subgraph, then run a _centrality_ algorithm to rerank the
most-referenced entities in the subgraph.
        """
        # TODO: visualize the walks -- the pathing does not seem to work?
        walks: set[ str ] = set(list(self.semantic_random_walk()))

        if debug:
            ic(walks)

        self.subgraph = self.work.ctx.erkg.subgraph(
            self.anchor_nodes.union(walks)
        )

        rank_iter: dict[ str, float ] = nx.pagerank(  # type: ignore
            self.subgraph,
            self.work.config["tr"]["tr_alpha"],
        ).items()

        for iri, rank in sorted(rank_iter, key = lambda x: x[1], reverse = True):  # type: ignore
            if debug:
                hit: dict = self.work.ctx.erkg.get_node(iri)  # type: ignore
                ic("tr", iri, rank, hit)  # type: ignore

            yield iri  # type: ignore


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
                for path in self.work.ctx.erkg.shortest_paths(pair[0], pair[1]):
                    if debug:
                        ic(path)

                    for iri in path:
                        if iri not in pair:
                            if debug:
                                hit: dict = self.work.ctx.erkg.get_node(iri)
                                ic("walk", iri, hit)

                            yield iri
            except nx.NetworkXNoPath:
                # ignore attempts when the source node is unreachable
                pass


    def find_chunk_neighbors (
        self,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ int ]:
        """
Find the neighboring chunks for each anchor node.
        """
        for iri in self.anchor_nodes:
            if debug:
                ic(iri)

            for neigh_iri in self.work.ctx.erkg.neighbors(iri):
                neigh_hit: dict = self.work.ctx.erkg.get_node(neigh_iri)

                if neigh_hit.get("kind") == NodeKind.CHUNK.value:
                    chunk_id: int = TextChunk.get_uid(neigh_iri)

                    if debug:
                        ic(neigh_iri, neigh_hit, chunk_id)

                    yield chunk_id


    def get_chunks_text (
        self,
        *,
        debug: bool = True,
        ) -> list[ str ]:
        """
Retrieve text for the combined list of chunks.
        """
        # TODO: should this be sorted, i.e., as a _reranking_ function?
        chunk_ids: str = ", ".join([ str(c_id) for c_id in self.rag_chunks.keys() ])  # pylint: disable=C0201

        if debug:
            ic(chunk_ids)

        chunks: list[ str ] = self.work.ctx.chunk_table.search().where(  # type: ignore
            f"uid IN ({chunk_ids})"
        ).select(
            [ "text" ]
        ).to_polars()["text"].to_list()

        return chunks
