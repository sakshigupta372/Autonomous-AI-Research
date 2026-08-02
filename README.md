# Autonomous AI Research Scientist

A multi-agent system that reads papers from arXiv, extracts a GraphRAG knowledge graph, compares approaches across papers, reflects on summary quality, attempts experiment reproduction, and autonomously searches for papers to fill knowledge gaps.

## Architecture (Phases 1–4)

```
Scout → Reader → Analyst → Writer → Critic ──┬→ Writer (reflection)
                                              └→ Comparator → Experimenter → PersistMemory
                                                                                    │
                                         GapPlanner ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ┘
                                              (autonomous loop, Phase 4)
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
| **GapPlanner** | 4 | Analyze graph gaps, propose next arXiv query |
| **Checkpoint** | 4 | SQLite session resume via `--session-id` / `--resume` |
| **Web UI** | 4 | FastAPI dashboard to browse outputs and trigger runs |

## Stack (fully free tier)

- **LLM:** [Groq](https://console.groq.com/keys) — free API key, Llama 3.3 70B
- **Embeddings:** ChromaDB local ONNX MiniLM — no API key
- **Sandbox:** Python subprocess with timeout (no Docker required)
- **Web:** FastAPI + Uvicorn

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

**Standard run:**

```bash
research-agent run --topic "GraphRAG knowledge graph retrieval" --max-papers 3
```

**Autonomous loop (Phase 4)** — after the first pass, GapPlanner finds gaps and searches again:

```bash
research-agent run --topic "GraphRAG knowledge graph retrieval" --max-papers 2 --autonomous --max-iterations 2
```

**Faster run (skip experiments):**

```bash
research-agent run --topic "transformer attention" --max-papers 3 --no-experiments
```

**Checkpoint / resume:**

```bash
research-agent run --topic "GraphRAG" --session-id abc123
research-agent sessions
research-agent run --topic "GraphRAG" --session-id abc123 --resume
```

**Web dashboard:**

```bash
research-agent web
# open http://127.0.0.1:8000
```

## Outputs

| Output | Location |
|--------|----------|
| Per-paper summary + concept graph + reproducibility | `outputs/summaries/{arxiv_id}.md` |
| Cross-paper comparison table | `outputs/summaries/comparison_{topic}.md` |
| Session log | `data/sessions/{topic}_{session_id}.json` |
| Checkpoint DB (resume) | `data/checkpoints.db` |
| Downloaded PDFs | `data/papers/` |

## Why scientific literature?

Academic papers have structured sections and open arXiv access — ideal for GraphRAG. The architecture is source-agnostic; swap the Scout agent to target PubMed, patents, or web docs.

## Roadmap

- **Phase 1** — Read + summarize + GraphRAG ✅
- **Phase 2** — Comparison + Critic reflection ✅
- **Phase 3** — Sandbox experiment reproduction ✅
- **Phase 4** — Autonomous loop + checkpoint/resume + web UI ✅
