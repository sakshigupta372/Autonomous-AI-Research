# Autonomous AI Research Scientist

A multi-agent system that reads papers from arXiv, extracts a GraphRAG knowledge graph, compares approaches across papers, reflects on summary quality, attempts experiment reproduction, and writes structured research reports.

## Architecture (Phases 1–3)

```
Scout -> Reader -> Analyst -> Writer -> Critic -+-> Writer (reflection loop)
                                               |
                                               +-> Comparator -> Experimenter -> PersistMemory
```

| Agent | Phase | Job |
|-------|-------|-----|
| **Scout** | 1 | Search arXiv, download PDFs |
| **Reader** | 1 | PDF parse, section-aware chunking |
| **Analyst** | 1 | GraphRAG entity/relation extraction |
| **Writer** | 1 | Grounded structured summaries |
| **Critic** | 2 | Rubric scoring + revision feedback |
| **Comparator** | 2 | Cross-paper method/dataset/metric table |
| **Experimenter** | 3 | Generate & run reproduction scripts in sandbox |

## Stack (fully free tier)

- **LLM:** [Groq](https://console.groq.com/keys) — free API key, Llama 3.3 70B
- **Embeddings:** ChromaDB local ONNX MiniLM — no API key
- **Sandbox:** Python subprocess with timeout (no Docker required)

## Setup

```bash
cd "Autonomous AI Research"
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
```

## Usage

Full pipeline (comparison + reflection + reproduction):

```bash
research-agent --topic "GraphRAG knowledge graph retrieval" --max-papers 3
```

Skip experiment reproduction for faster runs:

```bash
research-agent --topic "transformer attention mechanisms" --max-papers 3 --no-experiments
```

## Outputs

| Output | Location |
|--------|----------|
| Per-paper summary + concept graph + reproducibility | `outputs/summaries/{arxiv_id}.md` |
| Cross-paper comparison table | `outputs/summaries/comparison_{topic}.md` |
| Reflection & reproduction session log | `data/sessions/{topic}_{timestamp}.json` |
| Downloaded PDFs | `data/papers/` |
| Vector embeddings | `data/chroma/` |
| Knowledge graph + ingestion ledger | `data/graph.db` |

Re-running the same topic skips re-analyzing papers already in `data/graph.db`.

## Why scientific literature?

This MVP targets **academic papers** because they have structured sections (abstract, methods, results), open access via arXiv, and clear entities to graph (methods, datasets, metrics). The architecture is source-agnostic — the same pipeline can extend to PubMed, patents, or web docs by swapping the Scout agent.

## Project layout

```
src/research_agent/
├── main.py
├── config.py
├── state.py
├── graph.py
├── agents/       scout, reader, analyst, writer, critic, comparator, experimenter
├── tools/        arxiv_client, pdf_parser, graph_builder, sandbox
├── memory/       vector_store, graph_store
└── models/       paper, summary, comparison, critique, reproduction
```

## Roadmap

- **Phase 1** — Read + summarize + GraphRAG (done)
- **Phase 2** — Comparison + Critic reflection loop (done)
- **Phase 3** — Sandbox experiment reproduction (done)
- **Phase 4** — Autonomous research loop, web UI, checkpoint/resume (planned)
