"""NetworkX-based knowledge graph construction and export (GraphRAG MVP)."""

from __future__ import annotations

import networkx as nx

from research_agent.models.paper import PaperGraph


def new_graph() -> nx.MultiDiGraph:
    return nx.MultiDiGraph()


def merge_paper_graph(graph: nx.MultiDiGraph, paper_graph: PaperGraph) -> None:
    """Merge a per-paper entity/relation extraction into the global graph."""
    for entity in paper_graph.entities:
        node_id = f"{entity.type}:{entity.name}"
        if graph.has_node(node_id):
            graph.nodes[node_id]["papers"].add(paper_graph.paper_id)
        else:
            graph.add_node(
                node_id,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                papers={paper_graph.paper_id},
            )

    for relation in paper_graph.relations:
        source_candidates = [n for n in graph.nodes if graph.nodes[n]["name"] == relation.source]
        target_candidates = [n for n in graph.nodes if graph.nodes[n]["name"] == relation.target]
        if not source_candidates or not target_candidates:
            continue
        graph.add_edge(
            source_candidates[0],
            target_candidates[0],
            type=relation.type,
            description=relation.description,
            paper_id=paper_graph.paper_id,
        )


def subgraph_for_paper(graph: nx.MultiDiGraph, paper_id: str) -> nx.MultiDiGraph:
    """Return the subgraph of nodes/edges touched by a specific paper."""
    nodes = [n for n, data in graph.nodes(data=True) if paper_id in data.get("papers", set())]
    return graph.subgraph(nodes).copy()


def to_mermaid(graph: nx.MultiDiGraph, paper_id: str | None = None) -> str:
    """Render a graph (or a paper's subgraph) as a Mermaid flowchart."""
    target = subgraph_for_paper(graph, paper_id) if paper_id else graph
    lines = ["flowchart LR"]
    node_ids = {n: f"n{i}" for i, n in enumerate(target.nodes)}
    for node, data in target.nodes(data=True):
        label = data.get("name", node).replace('"', "'")
        lines.append(f'    {node_ids[node]}["{label}"]')
    for source, target_node, data in target.edges(data=True):
        rel_type = data.get("type", "")
        lines.append(f"    {node_ids[source]} -->|{rel_type}| {node_ids[target_node]}")
    return "\n".join(lines)


def to_json(graph: nx.MultiDiGraph) -> dict:
    """Serialize the graph to a JSON-friendly structure (hand-rolled to stay
    stable across networkx versions rather than relying on node_link_data).
    """
    nodes = [
        {
            "id": node_id,
            "name": data.get("name"),
            "type": data.get("type"),
            "description": data.get("description", ""),
            "papers": sorted(data.get("papers", set())),
        }
        for node_id, data in graph.nodes(data=True)
    ]
    edges = [
        {
            "source": source,
            "target": target,
            "type": data.get("type"),
            "description": data.get("description", ""),
            "paper_id": data.get("paper_id"),
        }
        for source, target, data in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def from_json(data: dict) -> nx.MultiDiGraph:
    """Deserialize a graph previously produced by to_json."""
    graph = new_graph()
    for node in data.get("nodes", []):
        graph.add_node(
            node["id"],
            name=node.get("name"),
            type=node.get("type"),
            description=node.get("description", ""),
            papers=set(node.get("papers", [])),
        )
    for edge in data.get("edges", []):
        graph.add_edge(
            edge["source"],
            edge["target"],
            type=edge.get("type"),
            description=edge.get("description", ""),
            paper_id=edge.get("paper_id"),
        )
    return graph
