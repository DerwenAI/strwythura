#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive visualization of a `NetworkX` graph based on `PyVis` to
generate an HTML file. Subclass this interface to integrate other
visualization libraries.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing

import pyvis  # type: ignore

from .context import DomainContext


class VisHTML:  # pylint: disable=R0902
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


    def gen_vis_html (  # pylint: disable=R0914
        self,
        domain_context: DomainContext,
        html_file: str,
        *,
        num_docs: int = 1,
        notebook: bool = False,
        ) -> None:
        """
Use `pyvis` to provide an interactive visualization of the graph layers.
        """
        kept_nodes: typing.Set[ int ] = set()

        pv_net: pyvis.network.Network = pyvis.network.Network(
            height = "900px",
            width = "100%",
            notebook = notebook,
            cdn_resources = "remote",
        )

        for node_id, attr in domain_context.vis_nodes(num_docs):
            kept_nodes.add(node_id)

            pv_net.add_node(
                node_id,
                label = attr["label"],
                title = attr["title"],
                color = attr["color"],
                size = attr["size"],
            )

        for src_node, dst_node, key in domain_context.sem_layer.edges(keys = True):
            if src_node in kept_nodes and dst_node in kept_nodes:
                pv_net.add_edge(
                    src_node,
                    dst_node,
                    title = key,
                )

            pv_net.toggle_physics(True)
            pv_net.show_buttons(filter_ = [ "physics" ])
            pv_net.save_graph(html_file)
