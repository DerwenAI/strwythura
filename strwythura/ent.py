#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the entity store.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

from collections import OrderedDict
import json
import pathlib

from arrowspace import ArrowSpaceBuilder, GraphLaplacian  # type: ignore  # pylint: disable=E0611
from icecream import ic
import gensim  # type: ignore
import numpy as np

from .elem import Entity, \
    de_token_span


class EntityStore:
    """
Manage the semantic graph embeddings with this domain context:
    vocabulary => taxonomy => thesaurus => ontology
    """

    def __init__ (
        self,
        config: dict,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config

        # manage the known entities
        self.entities: OrderedDict = OrderedDict()

        # entity UIDs used for embedding, etc.
        self.max_uid: int = 0

        # entity embeddings
        self.w2v_vectors: list[ list[ int ] ] = []
        self.w2v_model: gensim.models.Word2Vec | None = None


    def increment_uid (
        self,
        ) -> int:
        """
Increment the count of nodes.
        """
        uid: int = self.max_uid
        self.max_uid += 1

        return uid


    def get_decoder (
        self,
        ) -> dict[ int, Entity ]:
        """
Construct a dictionary to lookup entities by their `uid` value.
        """
        return {
            ent.uid: ent
            for ent in self.entities.values()
        }


    def encode_entity (
        self,
        ent: Entity,
        *,
        create: bool = False,
        ) -> Entity | None:
        """
Encode a known entity, indexed by its parsed lemma key in the entity
store, and set its IRI as a unique identifier in the ERKG.

This encoding serves as a UID for nodes in the semantic layer and
within vector representation for embeddings.

If a entity arrives with a duplicate `lemma_key` then promote the
one which comes from a higher priority source (and therefore more
info populated).

Return the entity which is stored.
        """
        if create:
            if ent.lemma_key not in self.entities:
                # add a new entity into the store
                ent.uid = self.increment_uid()
                ent.count = 1
                self.entities[ent.lemma_key] = ent

            elif ent.span.source < self.entities[ent.lemma_key].span.source:
                # replace previous entity with one from a a higher priority source
                ent.uid = self.entities[ent.lemma_key].uid
                ent.count = self.entities[ent.lemma_key].count + 1
                self.entities[ent.lemma_key] = ent

            else:
                # increment the incidence count
                self.entities[ent.lemma_key].count += 1

        return self.entities.get(ent.lemma_key)


    def load_json (
        self,
        store_path: pathlib.Path,
        ) -> None:
        """
De-serialize the entity definitions from a JSONL file, first clearing
any previous definitions.
        """
        self.entities = OrderedDict()

        with store_path.open("r", encoding = "utf-8") as fp:
            for line in fp:
                ent: Entity = Entity.model_validate(
                    json.loads(line),
                )

                self.entities[ent.lemma_key] = ent
                self.max_uid = max(self.max_uid, ent.uid + 1)  # type: ignore

        ic(self.max_uid, len(self.entities))


    def save_json (
        self,
        store_path: pathlib.Path,
        ) -> None:
        """
Serialize the entity definitions to a JSONL file.
        """
        with store_path.open("w",  encoding = "utf-8") as fp:
            for ent in self.entities.values():
                ent.span.span = de_token_span(ent.span.span)
                fp.write(json.dumps(ent.model_dump(mode = "json")))
                fp.write("\n")


    def embed_sequence (
        self,
        seq_vec: list[ int ],
        ) -> None:
        """
Build an embedding vector input for a sequence of entities from a
parsed sentence.
        """
        self.w2v_vectors.append(seq_vec)


    def load_vec (
        self,
        vec_path: pathlib.Path,
        ) -> None:
        """
De-serialize the entity embedding vectors from a text file,
overwriting any previous definitions.
        """
        with vec_path.open("r", encoding = "utf-8") as fp:
            self.w2v_vectors = []

            for line in fp:
                self.w2v_vectors.append(
                    [ int(x) for x in line.strip().split(",") ]
                )


    def save_vec (
        self,
        vec_path: pathlib.Path,
        ) -> None:
        """
Serialize the entity embedding vectors to a text file.
        """
        with vec_path.open("w", encoding = "utf-8") as fp:
            for vec in self.w2v_vectors:
                vec_rep: str = ",".join(map(str, vec))

                # filter null vectors
                if len(vec_rep) > 0:
                    fp.write(vec_rep)
                    fp.write("\n")


    def train_embeddings (
        self,
        ) -> None:
        """
Train a `gensim.Word2Vec` model for entity embeddings.
        """
        w2v_max: int = max([  # pylint: disable=R1728
            len(vec) - 1
            for vec in self.w2v_vectors
        ])

        # TODO: here's where we need to resolve once synonyms get
        # introduced into the thesaurus via curation
        w2v_vect: list[list[ str ]] = [
            [
                str(x)
                for x in vec
            ]
            for vec in self.w2v_vectors
        ]

        self.w2v_model = gensim.models.Word2Vec(
            sentences = w2v_vect,
            vector_size = 23,
            window = w2v_max,
            min_count = 1,
            sg = 1, # use skip-gram, not CBOW
        )


    def load_w2v (
        self,
        w2v_path: pathlib.Path,
        ) -> None:
        """
De-serialize the entity embeddings model from a file in `gensim.Word2Vec`
format.
        """
        self.w2v_model = gensim.models.Word2Vec.load(w2v_path.as_posix())


    def save_w2v (
        self,
        w2v_path: pathlib.Path,
        ) -> None:
        """
Serialize the entity embeddings model to a file in `gensim.Word2Vec`
format.
        """
        self.w2v_model.save(w2v_path.as_posix())  # type: ignore


    def build_aspace (
        self,
        w2v_model: gensim.models.Word2Vec,
        *,
        tau: float = 1.0,
        debug: bool = True,
        ) -> tuple[ ArrowSpaceBuilder, GraphLaplacian ]:
        """
Build an `ArrowSpace` computed signal graph and lambdas,
then compare with `gensim` similarity measures.
        """
        # extract the embedding vectors from a `gensim` model
        embed_vecs: list = []

        for uid, lemma_key in enumerate(self.entities.keys()):
            entity_key: str = str(uid)

            if entity_key in w2v_model.wv.index_to_key:
                embedding: list[ float ] = w2v_model.wv[entity_key]
                embed_vecs.append(embedding)

                if debug:
                    ic(uid, lemma_key)
                    print(embedding)

        # build an ArrowSpace with computed signal graph and lambdas
        aspace_params: dict = {
            "eps": 1.0,
            "k": 6,
            "topk": 6,
            "p": 2.0,
            "sigma": 1.0,
        }

        aspace, gl = ArrowSpaceBuilder.build(
            aspace_params,
            np.array(
                embed_vecs,
                dtype = np.float64,
            ),
        )

        # search comparable items
        # defaults: k = nitems, alpha = 0.9, beta = 0.1
        for query_uid in [ ent.uid for ent in self.entities.values() ]:
            query: np.ndarray = np.array(
                embed_vecs[query_uid],
                dtype = np.float64,
            )

            similar_words = w2v_model.wv.most_similar(
                str(query_uid),
                topn = 5,
            )

            if debug:
                ic(similar_words)

                for uid, sim_metric in aspace.search(query, gl, tau):
                    if uid != query_uid:
                        ic(uid, sim_metric)

        # return the ArrowSpace and Graph laplacian
        return aspace, gl
