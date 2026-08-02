"""Typer CLI entry point for the Autonomous AI Research Scientist."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from research_agent.config import settings
from research_agent.graph import run_research

app = typer.Typer(help="Autonomous AI Research Scientist CLI")
console = Console()


@app.command()
def run(
    topic: str = typer.Option(..., "--topic", "-t", help="Research topic or arXiv search query"),
    max_papers: int = typer.Option(
        settings.research_agent_max_papers, "--max-papers", "-n", help="Maximum number of papers to process"
    ),
    no_experiments: bool = typer.Option(
        False, "--no-experiments", help="Skip Phase 3 experiment reproduction (faster runs)"
    ),
) -> None:
    """Search arXiv, read papers, compare, reflect, reproduce experiments, and write reports."""
    if not settings.groq_api_key:
        console.print(
            "[red]GROQ_API_KEY is not set. Copy .env.example to .env and add a free key from "
            "https://console.groq.com/keys[/red]"
        )
        raise typer.Exit(code=1)

    console.print(f"[bold]Researching:[/bold] {topic} (max {max_papers} papers)")
    if no_experiments:
        console.print("[dim]Experiments disabled (--no-experiments)[/dim]")
    console.print(
        "[dim]Pipeline: Scout → Reader → Analyst → Writer → Critic → Comparator → "
        f"{'(skip Experimenter) → ' if no_experiments else 'Experimenter → '}PersistMemory[/dim]\n"
    )
    final_state = run_research(topic, max_papers, enable_experiments=not no_experiments)

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

    for path in final_state["report_paths"]:
        console.print(f"[green]Report written:[/green] {path}")

    if final_state.get("reflection_logs"):
        console.print("[cyan]Reflection log:[/cyan]")
        for entry in final_state["reflection_logs"]:
            console.print(f"  - {entry}")

    if final_state["errors"]:
        console.print("[yellow]Warnings/errors during run:[/yellow]")
        for err in final_state["errors"]:
            console.print(f"  - {err}")


if __name__ == "__main__":
    app()
