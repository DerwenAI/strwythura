#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Part 6: distill from the lexical graph to the resulting ERKG, and also
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

    work.ctx.ent_store.load_json(
        pathlib.Path(work.config["ent"]["store_path"]),
    )

    work.ctx.ent_store.load_vec(
        pathlib.Path(work.config["ent"]["vec_path"]),
    )

    work.ctx.lex.load_graph(
        pathlib.Path(work.config["nlp"]["lex_path"]),
    )

    work.ctx.erkg.load_graph(
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )

    work.ctx.open_vector_tables()
    work.load_parser()

    # distill graph elements from the lexical graph into the ERKG
    work.distill_knowledge_graph()

    # serialize the intermediate results
    work.ctx.lex.save_graph(
        pathlib.Path(work.config["nlp"]["lex_path"]),
    )

    work.ctx.erkg.save_graph(
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )

    # TODO: here's where we need to resolve synonyms, after synonyms
    # are introduced into the thesaurus through curation
    work.ctx.ent_store.train_embeddings()

    work.ctx.ent_store.save_w2v(
        pathlib.Path(work.config["ent"]["w2v_path"]),
    )

    # TODO: build an `ArrowSpace` computed signal graph and lambdas
    # then compare top-K results with `gensim`
    if False:  # pylint: disable=W0125
        aspace, gl = work.ctx.ent_store.build_aspace(
            work.ctx.ent_store.w2v_model,
        )


    # finally, report the performance profiler stats
    if profiling:
        prof.analyze()
