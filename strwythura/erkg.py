#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the ERKG knowledge graph, based on `NetworkX`.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import inspect
import json
import math
import pathlib
import sys
import traceback
import typing

from icecream import ic
import networkx as nx

from .elem import EntitySource, NodeKind, NODE_STYLES


class KnowledgeGraph:
    """
Represent the ERKG knowledge graph asset.
    """

    def __init__ (
        self,
        config: dict,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()


    ######################################################################
    ## serialization

    def load_graph (
        self,
        erkg_path: pathlib.Path,
        ) -> None:
        """
De-serialize a constructed KG from a JSON file represented in the
_node-link_ data format.
        """
        with erkg_path.open("r", encoding = "utf-8") as fp:
            self.graph = nx.node_link_graph(
                json.load(fp),
                edges = "edges",
            )


    def save_graph (
        self,
        erkg_path: pathlib.Path,
        ) -> None:
        """
Serialize the constructed KG as a JSON file represented in the
_node-link_ data format.

Aternatively this could be stored in a graph database.
        """
        with erkg_path.open("w", encoding = "utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(
                        self.graph,
                        edges = "edges",
                    ),
                    indent = 2,
                    sort_keys = True,
                )
            )


    ######################################################################
    ## graph construction

    def add_edge (  # pylint: disable=W0102
        self,
        src_iri: str,
        rel_iri: str,
        dst_iri: str,
        prob = 0.0,
        *,
        attrs: dict = {},
        update: bool = False,
        stop: bool = True,
        debug: bool = False,
        ) -> dict | None:
        """
Add an edge into the ERKG with required and optional properties.

Required properties: each edge must have an IRI as its `MultiGraph`
unique key, and a `prob` probability value.

Optional properties: specified as key/value pairs in the `attrs`
dictionary.
        """
        edge: tuple = ( src_iri, dst_iri, rel_iri, )
        pre_exist: bool = False

        # override conflicting settings
        if update:
            stop = False

        # test whether the edge IRI already exists in the ERKG?
        if self.graph.has_edge(*edge):
            pre_exist = True

            calframe: list = inspect.getouterframes(inspect.currentframe(), 2)
            caller: str = calframe[1][3]
            prev_attrs: dict = self.graph.edges[*edge]

            if debug | stop:
                print(f"dupe: {caller} {edge} {prob} {attrs}")
                print("PRE-EXISTING EDGE", prev_attrs)

            if stop:
                # if requested for debugging, stop the application
                sys.exit(-1)
            elif not update:
                # return the pre-existing edge data and do not update
                return prev_attrs

        # add an edge into the ERKG
        if not pre_exist:
            if debug:
                ic("ADD EDGE", edge, prob, attrs)

            self.graph.add_edge(
	        src_iri,
                dst_iri,
	        key = rel_iri,
	        prob = prob,
            )

        # set the optional edge attributes, if any
        if (update or not pre_exist) and len(attrs) > 0:
            nx.set_edge_attributes(
                self.graph,
                { edge: attrs },
            )

        return None


    def add_node (  # pylint: disable=W0102
        self,
        iri: str,
        kind: NodeKind,
        *,
        attrs: dict = {},
        update: bool = False,
        stop: bool = True,
        debug: bool = False,
        ) -> dict | None:
        """
Add a node into the ERKG with required and optional properties.

Required properties: each node must have an IRI as its unique
identifier, and a `NodeKind` value.

Optional properties: specified as key/value pairs in the `attrs`
dictionary.
        """
        pre_exist: bool = False

        # override conflicting settings
        if update:
            stop = False

        # test whether the node IRI already exists in the ERKG?
        if self.graph.has_node(iri):
            pre_exist = True

            calframe: list = inspect.getouterframes(inspect.currentframe(), 2)
            caller: str = calframe[1][3]
            prev_attrs: dict = self.graph.nodes[iri]

            if debug | stop:
                print(f"dupe: {caller} {iri} {kind}")
                print("PRE-EXISTING NODE", prev_attrs)

            if stop:
                # if requested for debugging, stop the application
                sys.exit(-1)
            elif not update:
                # return the pre-existing node data and do not update
                return prev_attrs

        # add a node into the ERKG
        if not pre_exist:
            if debug:
                ic("ADD NODE", iri, kind.value, attrs)

            self.graph.add_node(
                iri,
                kind = kind.value,
            )
        else:
            attrs["kind"] = kind.value

        # set the optional node attributes, if any
        if (update or not pre_exist) and len(attrs) > 0:

            nx.set_node_attributes(
                self.graph,
                { iri: attrs },
            )

        return None


    ######################################################################
    ## accessors for graph elements and views

    def get_node (
        self,
        iri: str,
        ) -> dict:
        """
Accessor method to get the properties of an ERKG node.
        """
        return self.graph.nodes[iri]


    def inbound_edges (
        self,
        dst_iri: str,
        *,
        attr: str = "weight",
        ) -> typing.Iterator[ tuple[ str, set[ str ], float ] ]:
        """
Iterator for the inbound connections of an ERKG node.
        """
        for src_iri, _, keys, val in self.graph.in_edges(  # type: ignore
            nbunch = dst_iri,
            data = attr,
            keys = True,
        ):
            yield src_iri, keys, val


    def neighbors (
        self,
        iri: str,
        ) -> typing.Iterator[ str ]:
        """
Iterator for the neighbors of an ERKG node.
        """
        try:
            yield from self.graph.neighbors(iri)
        except nx.exception.NetworkXError as nx_ex:
            # no neighbors
            ic(nx_ex)
        except Exception as ex:  # pylint: disable=W0718
            ic(ex)
            traceback.print_exc()


    def subgraph (
        self,
        iri_set: set[ str ],
        ) -> nx.Graph:
        """
Returns a subgraph view from the given node IRIs.
        """
        return self.graph.subgraph(iri_set)


    def shortest_paths (
        self,
        src_iri: str,
        dst_iri: str,
        ) -> typing.Iterator[ list[ str ] ]:
        """
Compute all shortest simple paths in the graph.
        """
        return nx.all_shortest_paths(self.graph, src_iri, dst_iri)


    ######################################################################
    ## interactive visualization

    def vis_nodes (
        self,
        num_docs: int,
        ) -> typing.Iterator[ tuple[ int, dict, ] ]:
        """
Iterator for the visualization attributes of nodes.
        """
        filter_iri: set[ str ] = set([ "sz:Entity", "sz:DataRecord", ])
        filter_lex: set[ str ] = set([ EntitySource.NC.value, EntitySource.LEX.value, ])

        for iri, attrs in self.graph.nodes(data = True):
            node: dict = {}

            if attrs.get("kind") == NodeKind.TAXONOMY.value and attrs.get("id") not in filter_iri:
                node["style"] = NODE_STYLES[EntitySource.TAXO]
                node["size"] = 5
                node["label"] = attrs.get("id")  # type: ignore
                node["title"] = attrs.get("text")  # type: ignore

            elif attrs.get("kind") == NodeKind.ENTITY.value and attrs.get("method") in [ EntitySource.ER.value, ]:
                node["style"] = NODE_STYLES[EntitySource.ER]
                node["size"] = round(50 * math.log(1.0 + math.sqrt(float(attrs.get("count"))) / num_docs))
                node["label"] = attrs.get("text")  # type: ignore
                node["title"] = attrs.get("lemma")  # type: ignore

            elif attrs.get("kind") == NodeKind.ENTITY.value and attrs.get("method") in [ EntitySource.NER.value, ]:
                node["style"] = NODE_STYLES[EntitySource.NER]
                node["size"] = round(20 * math.log(1.0 + math.sqrt(float(attrs.get("count"))) / num_docs))
                node["label"] = attrs.get("text")  # type: ignore
                node["title"] = attrs.get("lemma")  # type: ignore

            elif attrs.get("kind") == NodeKind.ENTITY.value and attrs.get("label") not in filter_lex:
                ## NOTE: not included yet -- too much noise
                continue

            elif attrs.get("kind") == NodeKind.DATAREC.value:
                ## NOTE: the `iri` value for a node_id would be displayed in PyVis -- too much noise
                continue

            else:
                continue

            yield iri, node


    def vis_edges (
        self,
        ) -> typing.Iterator[ tuple[ int, int, str, ] ]:
        """
Iterator for the visualization attributes of edges.
        """
        filter_rel: set[ str ] = set([ "strw:co_occurs_with", ])

        for src_iri, dst_iri, key in self.graph.edges(keys = True):
            if key not in filter_rel:
                yield src_iri, dst_iri, key
