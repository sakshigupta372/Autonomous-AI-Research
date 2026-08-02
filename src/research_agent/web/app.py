"""FastAPI web dashboard for browsing research outputs (Phase 4)."""

from __future__ import annotations

import json
import re
import threading

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from research_agent.checkpoint import list_sessions, new_session_id
from research_agent.config import settings
from research_agent.graph import run_research

app = FastAPI(title="Autonomous AI Research Scientist", version="0.3.0")

_run_status: dict[str, str] = {}


class ResearchRequest(BaseModel):
    topic: str
    max_papers: int = Field(default=3, ge=1, le=10)
    autonomous: bool = False
    max_autonomous_iterations: int = Field(default=2, ge=0, le=5)
    no_experiments: bool = False


def _list_summaries() -> list[dict]:
    return [
        {"arxiv_id": path.stem, "filename": path.name}
        for path in sorted(settings.outputs_dir.glob("*.md"))
        if not path.name.startswith("comparison_")
    ]


def _list_comparisons() -> list[dict]:
    return [
        {"filename": p.name, "topic_slug": p.stem.replace("comparison_", "")}
        for p in sorted(settings.outputs_dir.glob("comparison_*.md"))
    ]


def _list_session_logs() -> list[dict]:
    logs = []
    for path in sorted(settings.sessions_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logs.append({"filename": path.name, "topic": data.get("topic"), "timestamp": data.get("timestamp")})
        except json.JSONDecodeError:
            logs.append({"filename": path.name, "topic": None, "timestamp": None})
    return logs


def _run_pipeline(session_id: str, req: ResearchRequest) -> None:
    _run_status[session_id] = "running"
    try:
        run_research(
            topic=req.topic,
            max_papers=req.max_papers,
            enable_experiments=not req.no_experiments,
            session_id=session_id,
            autonomous=req.autonomous,
            max_autonomous_iterations=req.max_autonomous_iterations,
        )
        _run_status[session_id] = "completed"
    except Exception as exc:  # noqa: BLE001
        _run_status[session_id] = f"failed: {exc}"


def _dashboard_html() -> str:
    summaries = _list_summaries()
    comparisons = _list_comparisons()
    sessions = _list_session_logs()
    checkpoints = list_sessions()

    summary_rows = "".join(
        f'<li><a href="/summaries/{s["arxiv_id"]}">{s["arxiv_id"]}</a></li>' for s in summaries
    ) or "<li>No summaries yet</li>"
    comparison_rows = "".join(
        f'<li><a href="/comparisons/{c["topic_slug"]}">{c["filename"]}</a></li>' for c in comparisons
    ) or "<li>No comparisons yet</li>"
    session_rows = "".join(
        f'<li>{s["filename"]}</li>' for s in sessions[:10]
    ) or "<li>No sessions yet</li>"
    checkpoint_rows = "".join(f"<li><code>{sid}</code></li>" for sid in checkpoints[:10]) or "<li>None</li>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Research Scientist</title>
<style>
body {{ font-family: system-ui,sans-serif; max-width:960px; margin:2rem auto; padding:0 1rem; }}
.card {{ border:1px solid #ddd; border-radius:8px; padding:1rem; margin-bottom:1rem; }}
input,button {{ padding:0.5rem; margin:0.25rem 0; }}
pre {{ background:#f6f8fa; padding:1rem; overflow-x:auto; white-space:pre-wrap; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
</style></head><body>
<h1>Autonomous AI Research Scientist</h1>
<div class="card">
<h2>Start research</h2>
<form id="runForm">
<label>Topic<br><input id="topic" size="60" required></label><br>
<label>Max papers <input id="max_papers" type="number" value="3" min="1" max="10"></label><br>
<label><input id="autonomous" type="checkbox"> Autonomous loop</label><br>
<label><input id="no_experiments" type="checkbox"> Skip experiments</label><br>
<button type="submit">Run</button>
</form>
<pre id="runResult">Ready.</pre>
</div>
<div class="grid">
<div class="card"><h3>Summaries</h3><ul>{summary_rows}</ul></div>
<div class="card"><h3>Comparisons</h3><ul>{comparison_rows}</ul></div>
<div class="card"><h3>Sessions</h3><ul>{session_rows}</ul></div>
<div class="card"><h3>Checkpoints</h3><ul>{checkpoint_rows}</ul></div>
</div>
<script>
document.getElementById('runForm').onsubmit = async (e) => {{
  e.preventDefault();
  const body = {{
    topic: document.getElementById('topic').value,
    max_papers: Number(document.getElementById('max_papers').value),
    autonomous: document.getElementById('autonomous').checked,
    no_experiments: document.getElementById('no_experiments').checked,
  }};
  const res = await fetch('/api/research', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body)}});
  const data = await res.json();
  document.getElementById('runResult').textContent = JSON.stringify(data,null,2);
  if (data.session_id) {{
    const poll = setInterval(async () => {{
      const st = await fetch('/api/status/' + data.session_id);
      const info = await st.json();
      document.getElementById('runResult').textContent = JSON.stringify(info,null,2);
      if (info.status !== 'running' && info.status !== 'queued') clearInterval(poll);
    }}, 3000);
  }}
}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _dashboard_html()


@app.get("/api/summaries")
def api_summaries() -> list[dict]:
    return _list_summaries()


@app.get("/summaries/{arxiv_id}", response_class=HTMLResponse)
def get_summary(arxiv_id: str) -> str:
    path = settings.outputs_dir / f"{arxiv_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    content = path.read_text(encoding="utf-8")
    escaped = content.replace("&", "&amp;").replace("<", "&lt;")
    return f"<!DOCTYPE html><html><body><pre>{escaped}</pre><a href='/'>Back</a></body></html>"


@app.get("/comparisons/{topic_slug}", response_class=HTMLResponse)
def get_comparison(topic_slug: str) -> str:
    path = settings.outputs_dir / f"comparison_{topic_slug}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Comparison not found")
    content = path.read_text(encoding="utf-8").replace("&", "&amp;").replace("<", "&lt;")
    return f"<!DOCTYPE html><html><body><pre>{content}</pre><a href='/'>Back</a></body></html>"


@app.post("/api/research")
def start_research(req: ResearchRequest, background_tasks: BackgroundTasks) -> dict:
    if not settings.groq_api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not configured")
    session_id = new_session_id()
    _run_status[session_id] = "queued"
    background_tasks.add_task(_run_pipeline, session_id, req)
    return {"session_id": session_id, "status": "queued", "topic": req.topic}


@app.get("/api/status/{session_id}")
def research_status(session_id: str) -> dict:
    return {"session_id": session_id, "status": _run_status.get(session_id, "unknown")}


@app.get("/api/sessions")
def api_sessions() -> list[dict]:
    return _list_session_logs()
