#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Part 6: use `PyVis` to generate an HTML page for interactive
visualization of the ERKG.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib

from strwythura import Profiler, VisHTML, Workflow


if __name__ == "__main__":
    # start the performance profiling -
    profiling: bool = True # False

    if profiling:
        prof: Profiler = Profiler()


    # instantiate and configure our workflow manager
    work: Workflow = Workflow(
        config_path = pathlib.Path("config.toml"),
    )

    work.ctx.erkg.load_graph(
        pathlib.Path(work.config["erkg"]["erkg_path"]),
    )

    domain: dict = json.load(
        pathlib.Path("domain.json").open("r", encoding = "utf-8")
    )

    num_docs: int = len(domain["sources"]["content"])

    # generate HTML for interactive visualization of the ERKG
    vis: VisHTML = VisHTML()
    vis.set_config(work.config)

    vis.gen_vis_html(
        pathlib.Path(work.config["vis"]["html_path"]),
        work.ctx.erkg.vis_nodes(num_docs),
        work.ctx.erkg.vis_edges(),
        work.config["vis"]["html_height"],
        work.config["vis"]["html_width"],
    )


    # finally, report the performance profiler stats
    if profiling:
        prof.analyze()
