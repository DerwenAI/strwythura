#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Part 5: distill from the lexical graph to the resulting ERKG, and also
build an _entities embeddings_ model.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import pathlib

from strwythura import Profiler, Workflow


if __name__ == "__main__":
    # start the performance profiling -
    profiling: bool = True # False

    if profiling:
        prof: Profiler = Profiler()


    # instantiate and configure our workflow manager
    work: Workflow = Workflow(
        config_path = pathlib.Path("config.toml"),
    )

    # de-serialize assets from previous steps
    work.thesaurus.load_source(
        pathlib.Path(work.config["sz"]["thesaurus_path"]),
        format = "turtle",
    )

    work.domain_ctx.ent_store.load_json(
        pathlib.Path(work.config["ent"]["store_path"]),
    )

    work.domain_ctx.ent_store.load_vec(
        pathlib.Path(work.config["ent"]["vec_path"]),
    )

    work.domain_ctx.lexical.load_graph(
        pathlib.Path(work.config["nlp"]["lex_path"]),
    )

    work.domain_ctx.load_erkg(
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )

    work.domain_ctx.open_vector_tables()
    work.load_parser()

    # distill graph elements from the lexical graph into the ERKG
    work.distill_knowledge_graph(
        pathlib.Path(work.config["nlp"]["lex_path"]),
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )

    # serialize the intermediate results
    work.domain_ctx.lexical.save_graph(
        pathlib.Path(work.config["nlp"]["lex_path"]),
    )

    work.domain_ctx.save_erkg(
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )

    # TODO: here's where we need to resolve synonyms, after synonyms
    # are introduced into the thesaurus through curation
    work.domain_ctx.ent_store.train_embeddings()

    work.domain_ctx.ent_store.save_w2v(
        pathlib.Path(work.config["ent"]["w2v_path"]),
    )

    # TODO: build an `ArrowSpace` computed signal graph and lambdas
    # then compare top-K results with `gensim`
    if False:
        aspace, gl = work.domain_ctx.ent_store.build_aspace(
            work.domain_ctx.ent_store.w2v_model,
        )


    # finally, report the performance profiler stats
    if profiling:
        prof.analyze()
