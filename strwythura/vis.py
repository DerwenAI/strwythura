#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive visualization of a `NetworkX` graph based on `PyVis` to
generate an HTML file. Subclass this interface to integrate other
visualization libraries.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import pathlib
import typing

import pyvis  # type: ignore

from .resources import PYVIS_JINJA_TEMPLATE


class VisHTML:
    """
Implementation for `PyVis` to generate an HTML visualization.
    """

    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = {}


    def set_config (
        self,
        config: dict,
        ) -> None:
        """
Accessor method to configure -- part of a design pattern to make the
iteractive visualization more "pluggable", i.e., to be subclassed and
customized for other visualization libraries.
        """
        self.config = config


    def gen_vis_html (
        self,
        html_path: pathlib.Path,
        node_iter: typing.Iterator[ tuple[ int, dict, ] ],
        edge_iter: typing.Iterator[ tuple[ int, int, str, ] ],
        height: str,
        width: str,
        *,
        notebook: bool = False,
        cdn_resources: str = "remote",
        ) -> None:
        """
Use `pyvis` to provide an interactive visualization of the graph layers.
        """
        pv_net: pyvis.network.Network = pyvis.network.Network(
            height = height,
            width = width,
            notebook = notebook,
            cdn_resources = cdn_resources,
        )

        kept_nodes: set[ int ] = set()

        for iri, attrs in node_iter:
            kept_nodes.add(iri)

            pv_net.add_node(
                iri,
                label = attrs["label"],
                title = attrs["title"],
                size = attrs["size"],
                color = attrs["style"].color,
                shape = attrs["style"].shape,
            )

        for src_iri, dst_iri, key in edge_iter:
            if src_iri in kept_nodes and dst_iri in kept_nodes:
                pv_net.add_edge(
                    src_iri,
                    dst_iri,
                    title = key,
                )

        pv_net.toggle_physics(True)
        pv_net.show_buttons(filter_ = [ "physics" ])

        pv_net.save_graph(html_path.as_posix())


    def rebuild_html (
        self,
        domain: dict,
        load_path: pathlib.Path,
        save_path: pathlib.Path,
        ) -> None:
        """
Rebuild the HTML app which `pyvis` generated, restructuring the UI:

  - less about embedding in Jypter notebook cells
  - more about `vis.js` controls in an HTML standalone app
        """
        node_frag: str = "nodes = new vis.DataSet("
        edge_frag: str = "edges = new vis.DataSet("

        nodes: str = "null;"
        edges: str = "null;"

        with load_path.open() as fp:
            for line in fp:
                line = line.strip()

                if line.startswith(node_frag):
                    nodes = line.replace(node_frag, "").rstrip(");")

                if line.startswith(edge_frag):
                    edges = line.replace(edge_frag, "").rstrip(");")

        with open(save_path, "w", encoding = "utf-8") as fp:
            fp.write(PYVIS_JINJA_TEMPLATE.render(
                domain = domain,
                nodes = nodes,
                edges = edges,
            ))
