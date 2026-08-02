"""Typer CLI entry point for the Autonomous AI Research Scientist."""

from __future__ import annotations

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from research_agent.checkpoint import list_sessions
from research_agent.config import settings
from research_agent.graph import run_research

app = typer.Typer(help="Autonomous AI Research Scientist CLI")
console = Console()


def _print_results(final_state: dict) -> None:
    table = Table(title="Papers Processed")
    table.add_column("arXiv ID")
    table.add_column("Title")
    for paper in final_state["papers"]:
        table.add_row(paper.arxiv_id, paper.title)
    console.print(table)

    if final_state.get("critiques"):
        critique_table = Table(title="Critic Scores")
        critique_table.add_column("Paper")
        critique_table.add_column("Overall")
        critique_table.add_column("Revised?")
        for c in final_state["critiques"]:
            critique_table.add_row(c.paper_id, f"{c.overall}/10", "yes" if c.needs_revision else "no")
        console.print(critique_table)

    if final_state.get("reproductions"):
        repro_table = Table(title="Reproducibility")
        repro_table.add_column("Paper")
        repro_table.add_column("Score")
        repro_table.add_column("Success")
        for r in final_state["reproductions"]:
            repro_table.add_row(r.paper_id, f"{r.reproducibility_score:.0%}", str(r.success))
        console.print(repro_table)

    paths = final_state.get("all_report_paths") or final_state.get("report_paths", [])
    for path in paths:
        console.print(f"[green]Report written:[/green] {path}")

    if final_state.get("reflection_logs"):
        console.print("[cyan]Reflection log:[/cyan]")
        for entry in final_state["reflection_logs"]:
            console.print(f"  - {entry}")

    if final_state.get("gap_queries"):
        console.print("[cyan]Autonomous search queries:[/cyan]")
        for q in final_state["gap_queries"]:
            console.print(f"  - {q}")

    if final_state.get("session_id"):
        console.print(f"[cyan]Checkpoint session id:[/cyan] {final_state['session_id']}")

    if final_state["errors"]:
        console.print("[yellow]Warnings/errors during run:[/yellow]")
        for err in final_state["errors"]:
            console.print(f"  - {err}")


@app.command()
def run(
    topic: str = typer.Option(..., "--topic", "-t", help="Research topic or arXiv search query"),
    max_papers: int = typer.Option(
        settings.research_agent_max_papers, "--max-papers", "-n", help="Maximum number of papers to process"
    ),
    no_experiments: bool = typer.Option(
        False, "--no-experiments", help="Skip Phase 3 experiment reproduction (faster runs)"
    ),
    autonomous: bool = typer.Option(
        False, "--autonomous", help="Phase 4: keep searching arXiv to fill knowledge graph gaps"
    ),
    max_iterations: int = typer.Option(
        settings.research_agent_max_autonomous_iterations,
        "--max-iterations",
        help="Extra autonomous search rounds after the first full pass",
    ),
    session_id: str = typer.Option("", "--session-id", help="Checkpoint session id (auto-generated if empty)"),
    resume: bool = typer.Option(False, "--resume", help="Resume a previous checkpointed session"),
) -> None:
    """Search arXiv, read papers, compare, reflect, reproduce, and optionally loop autonomously."""
    if not settings.groq_api_key:
        console.print(
            "[red]GROQ_API_KEY is not set. Copy .env.example to .env and add a free key from "
            "https://console.groq.com/keys[/red]"
        )
        raise typer.Exit(code=1)

    if resume and not session_id:
        console.print("[red]--resume requires --session-id[/red]")
        raise typer.Exit(code=1)

    if not resume:
        console.print(f"[bold]Researching:[/bold] {topic} (max {max_papers} papers)")
    else:
        console.print(f"[bold]Resuming session:[/bold] {session_id}")

    if autonomous:
        console.print(f"[dim]Autonomous mode: up to {max_iterations} extra search iteration(s)[/dim]")
    if no_experiments:
        console.print("[dim]Experiments disabled (--no-experiments)[/dim]")

    console.print(
        "[dim]Pipeline: Scout → Reader → Analyst → Writer → Critic → Comparator → "
        f"{'(skip Experimenter) → ' if no_experiments else 'Experimenter → '}PersistMemory"
        f"{' → GapPlanner ↺' if autonomous else ''}[/dim]\n"
    )

    final_state = run_research(
        topic=topic,
        max_papers=max_papers,
        enable_experiments=not no_experiments,
        session_id=session_id or None,
        autonomous=autonomous,
        max_autonomous_iterations=max_iterations,
        resume=resume,
    )
    _print_results(final_state)


@app.command()
def sessions() -> None:
    """List checkpoint session ids available for --resume."""
    ids = list_sessions()
    if not ids:
        console.print("[dim]No checkpoint sessions found.[/dim]")
        return
    for sid in ids:
        console.print(sid)


@app.command()
def web(
    host: str = typer.Option(settings.research_agent_web_host, "--host"),
    port: int = typer.Option(settings.research_agent_web_port, "--port"),
) -> None:
    """Launch the Phase 4 web dashboard."""
    console.print(f"[bold]Dashboard:[/bold] http://{host}:{port}")
    uvicorn.run("research_agent.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
