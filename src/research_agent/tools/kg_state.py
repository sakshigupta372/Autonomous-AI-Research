"""Helpers to keep knowledge_graph checkpoint-serializable (dict, not NetworkX)."""

from __future__ import annotations

import networkx as nx

from research_agent.tools import graph_builder


def as_graph(data: dict | nx.MultiDiGraph | None) -> nx.MultiDiGraph:
    if isinstance(data, nx.MultiDiGraph):
        return data
    if isinstance(data, dict) and data.get("nodes") is not None:
        return graph_builder.from_json(data)
    return graph_builder.new_graph()


def as_dict(graph: nx.MultiDiGraph) -> dict:
    return graph_builder.to_json(graph)
