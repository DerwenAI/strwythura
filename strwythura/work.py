#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the workflow components and dependency injection.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import logging
import os
import pathlib
import tomllib
import typing

from icecream import ic
from sz_semantics import Thesaurus
import networkx as nx

from .ctx import DomainContext, TextChunk
from .elem import Entity, NodeKind, STRW_PREFIX
from .ent import EntityStore
from .lex import LexicalGraph
from .nlp import Parser
from .scrape import Scraper


class Workflow:
    """
Manage multiple workflow phases.
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
        logger: logging.Logger = logging.getLogger(__name__)
        logging.basicConfig(level = logging.WARNING) # DEBUG

        # disable noisy logging
        logging.disable(logging.ERROR)
        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        with config_path.open("rb") as fp:
            self.config: dict = tomllib.load(fp)

        # dependency injection:
        #  `DomainContext` -- subclass for different domain contexts
        #  `Parser` -- subclass for different languages and dialects

        self.thesaurus: Thesaurus = Thesaurus()
        self.scraper: Scraper = Scraper(self.config)

        self.ctx: DomainContext = self.load_class(
            self.config["ctx"]["domain_class"],
            self.config,
            self.thesaurus,
            EntityStore(self.config),
            LexicalGraph(self.config),
        )

        self.parser: Parser | None = None


    @classmethod
    def load_class (
        cls,
        class_name: str,
        *args,
        ) -> typing.Any:
        """
Load a subclass, as a means of "plug-in" architecture for handling
different domain contexts, different languages and dialects, etc.
        """
        return globals()[class_name](*args)


    def load_parser (
        self,
        ) -> None:
        """
Load the parser, which includes the run-time cost of loading large-ish
models used in the `spaCy` NLP pipeline.
        """
        self.parser = self.load_class(
            self.config["nlp"]["parser_class"],
            self.config,
            self.ctx,
        )


    ######################################################################
    ## Part 2

    def populate_semantic_layer (
        self,
        export_path: pathlib.Path,
        domain_path: pathlib.Path,
        *,
        language: str = "en",
        ) -> None:
        """
Populate the semantic layer from the results of entity resolution.
        """
        self.thesaurus.load_source(self.thesaurus.DOMAIN_TTL)
        self.thesaurus.load_source(domain_path)

        # parse the JSON exported from Senzing ER to populate
        # a domain-specific thesaurus
        with export_path.open("r", encoding = "utf-8") as fp_json:
            for line in fp_json:
                for rdf_frag in self.thesaurus.parse_iter(line, language = language):
                    self.thesaurus.rdf_graph.parse(
                        data = self.thesaurus.RDF_PREAMBLE + rdf_frag,
                        format = "turtle",
                    )


    def build_graph_backbone (
        self,
        ) -> None:
        """
Reform the `RDFlib` semantic graph => `NetworkX` property graph
as a "backbone" for structuring the knowledge graph, prior to
entity linking from unstructured sources.
        """
        self.ctx.promote_taxo_nodes()
        self.ctx.promote_er_nodes(self.parser)
        self.ctx.promote_data_nodes()
        self.ctx.promote_er_edges()


    ######################################################################
    ## Part 3

    def crawl_chunk_parse (
        self,
        content_sources: list[ str ],
        *,
        debug: bool = False,
        ) -> None:
        """
For each of the given URLs:

  - load the document's HTML, through a cache
  - extract the document text as a sequence of paragraphs
  - create a `TextChunk` for each paragraph
  - add the chunk and its embedding to the vector store
  - parse the text in each chunk, linking into the graph
        """
        for url in content_sources:
            sent_id: int = 0

            ic(url)

            # add each text chunk and its embedding to the vector store
            for chunk_text in self.scraper.scrape_html(url):
                chunk: TextChunk = self.ctx.add_chunk(
                    url,
                    sent_id,
                    chunk_text,
                )

                # parse the text chunk
                num_sent: int = self.parser.parse_chunk(
                    chunk.uid,
                    chunk.text,
                    debug = debug,
                )

                if debug | True:
                    ic(chunk, num_sent)

                ## TODO: debugging
                if chunk.uid == 51:
                    with open("debug.json", "w", encoding = "utf-8") as fp:
                        fp.write(json.dumps(chunk.model_dump(mode = "json")))

                sent_id += num_sent


    ######################################################################
    ## Part 5

    def distill_knowledge_graph (
        self,
        lex_path: pathlib.Path,
        erkg_path: pathlib.Path,
        ) -> None:
        """
Finalize construction and serialization of the knowledge graph after
entity linking:

  - run _textrank_ on the lexical graph to rank entities by weight and count
  - calculate conditional probability of entity _co-occurrence_ within a sentence
  - promote distilled entities into the knowledge graph
  - cross-link entities and chunks within the knowledge graph
  - serialize both graphs for later use in inference
        """
        # run the `textrank` algorithm, approximating a hypervolume of
        # the `weight` and `count` to normalize as a ranking, then
        # update the rank values in both the lexical graph and the
        # entity store
        for _, row in self.ctx.lex.run_textrank().iterrows():
            node_id: int = row["node_id"]
            rank: float = round(float(row["rank"]), 5)

            nx.set_node_attributes(
                self.ctx.lex.lex_graph,
                { node_id : rank },
                "rank",
            )

            lemma_key: str = self.ctx.lex.lex_graph.nodes[node_id]["lemma_key"]
            ent: Entity = self.ctx.ent_store.entities[lemma_key]
            ent.rank = rank

        # calculate the conditional probability for each pair of
        # entities which co-occur in a sentence, then represent
        # in the `LexicalGraph` using semantic relations
        self.ctx.co_occur_entities()

        # promote distilled entities into the knowledge graph
        self.ctx.promote_ner_nodes()

        # cross-link chunks with entities
        for ent in self.ctx.ent_store.entities.values():
            for ent_inst in ent.inst:
                self.ctx.erkg.add_edge(
                    ent.node_id,
                    f"chunk_{ent_inst.chunk_id}",
                    key = f"{STRW_PREFIX}within_chunk",
                    prob = 1.0,
                    weight = ent.rank,
                    sent_id = ent_inst.sent_id,
                )


    ######################################################################
    ## Part 6

    def load_assets (
        self,
        ) -> None:
        """
De-serialize assets from the previous steps.
        """
        self.thesaurus.load_source(
            pathlib.Path(self.config["sz"]["thesaurus_path"]),
            format = "turtle",
        )

        self.ctx.ent_store.load_json(
            pathlib.Path(self.config["ent"]["store_path"]),
        )

        self.ctx.ent_store.load_w2v(
            pathlib.Path(self.config["ent"]["w2v_path"]),
        )

        self.ctx.lex.load_graph(
            pathlib.Path(self.config["nlp"]["lex_path"]),
        )

        self.ctx.load_erkg(
            pathlib.Path(self.config["erkg"]["erkg_path"]),
        )
