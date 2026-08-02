# Autonomous AI Research Scientist

A multi-agent system that reads papers from arXiv, extracts a GraphRAG knowledge graph, and writes structured research summaries — the Phase 1 MVP of a larger autonomous research pipeline (see the project plan for later phases: comparison, reflection, and experiment reproduction).

## Architecture

```
Scout -> Reader -> Analyst -> Writer -> PersistMemory
```

- **Scout** (`src/research_agent/agents/scout.py`) — searches arXiv for a topic and downloads PDFs, skipping papers already ingested (long-term memory).
- **Reader** (`src/research_agent/agents/reader.py`) — extracts text from each PDF and splits it into section-aware chunks.
- **Analyst** (`src/research_agent/agents/analyst.py`) — uses an LLM (Groq) with structured output to extract entities (methods, datasets, metrics, claims, limitations) and relations, then merges them into a NetworkX knowledge graph (GraphRAG).
- **Writer** (`src/research_agent/agents/writer.py`) — uses an LLM (Groq) with structured output, grounded in the paper's chunks and its subgraph, to produce a `ResearchSummary`.
- **PersistMemory** (in `src/research_agent/graph.py`) — embeds chunks into ChromaDB using a free local embedding model, persists the knowledge graph and ingested-paper records to SQLite, and renders Markdown reports with an embedded Mermaid concept graph.

## Stack (fully free tier)

- **LLM:** [Groq](https://console.groq.com/keys) — free API key, no credit card required, fast Llama models.
- **Embeddings:** ChromaDB's bundled local ONNX MiniLM model — no API key, runs on CPU, downloads once (~80MB) on first use.

## Setup

```bash
cd "Autonomous AI Research"
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
# then edit .env and set GROQ_API_KEY=gsk_... (get a free key at https://console.groq.com/keys)
```

## Usage

```bash
research-agent --topic "GraphRAG for scientific literature" --max-papers 3
```

Or without installing the console script:

```bash
python -m research_agent.main --topic "GraphRAG for scientific literature" --max-papers 3
```

Generated reports are written to `outputs/summaries/{arxiv_id}.md`. Downloaded PDFs live in `data/papers/`, vector embeddings in `data/chroma/`, and the persisted knowledge graph plus ingestion record in `data/graph.db` (SQLite).

Re-running the same topic will skip re-downloading and re-analyzing papers already recorded in `data/graph.db`.

## Project layout

```
src/research_agent/
├── main.py             # Typer CLI entry point
├── config.py           # Settings from environment / .env
├── state.py            # LangGraph ResearchState
├── graph.py            # LangGraph workflow wiring
├── agents/             # Scout, Reader, Analyst, Writer
├── tools/               # arXiv client, PDF parser, graph builder
├── memory/              # ChromaDB vector store, SQLite graph store
└── models/              # Pydantic schemas (Paper, Chunk, Entity, Summary, ...)
```

## Roadmap

Phase 1 (this MVP) covers read + summarize. Later phases (see project plan) add multi-paper comparison, a Critic agent with a reflection loop, and an Experimenter agent that reproduces experiments in a sandboxed environment.
