#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive visualization of a `NetworkX` graph based on `PyVis`
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import math

import networkx as nx
import pyvis  # type: ignore


def gen_pyvis (
    graph: nx.Graph,
    html_file: str,
    *,
    num_docs: int = 1,
    notebook: bool = False,
    ) -> None:
    """
Use `pyvis` to provide an interactive visualization of the graph layers.
    """
    pv_net: pyvis.network.Network = pyvis.network.Network(
        height = "900px",
        width = "100%",
        notebook = notebook,
        cdn_resources = "remote",
    )

    for node_id, node_attr in graph.nodes(data = True):
        if node_attr.get("kind") == "Entity":
            color: str = "hsla(65, 46%, 58%, 0.80)"
            size: int = round(20 * math.log(1.0 + math.sqrt(float(node_attr.get("count"))) / num_docs))  # type: ignore # pylint: disable=C0301
            label: str = node_attr.get("text")  # type: ignore
            title: str = node_attr.get("key")  # type: ignore
        else:
            color = "hsla(306, 45%, 57%, 0.95)"
            size = 5
            label = node_id
            title = node_attr.get("url")  # type: ignore

        pv_net.add_node(
            node_id,
            label = label,
            title = title,
            color = color,
            size = size,
        )

    for src_node, dst_node, edge_attr in graph.edges(data = True):
        pv_net.add_edge(
            src_node,
            dst_node,
            title = edge_attr.get("rel"),
        )

        pv_net.toggle_physics(True)
        pv_net.show_buttons(filter_ = [ "physics" ])
        pv_net.save_graph(html_file)
