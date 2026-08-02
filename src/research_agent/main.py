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
) -> None:
    """Search arXiv, read papers, build a knowledge graph, and write research summaries."""
    if not settings.groq_api_key:
        console.print(
            "[red]GROQ_API_KEY is not set. Copy .env.example to .env and add a free key from "
            "https://console.groq.com/keys[/red]"
        )
        raise typer.Exit(code=1)

    console.print(f"[bold]Researching:[/bold] {topic} (max {max_papers} papers)")
    final_state = run_research(topic, max_papers)

    table = Table(title="Papers Processed")
    table.add_column("arXiv ID")
    table.add_column("Title")
    for paper in final_state["papers"]:
        table.add_row(paper.arxiv_id, paper.title)
    console.print(table)

    for path in final_state["report_paths"]:
        console.print(f"[green]Report written:[/green] {path}")

    if final_state["errors"]:
        console.print("[yellow]Warnings/errors during run:[/yellow]")
        for err in final_state["errors"]:
            console.print(f"  - {err}")


if __name__ == "__main__":
    app()
