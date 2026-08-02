"""Terminal progress logging for long-running pipeline steps."""

from __future__ import annotations

import sys

from rich.console import Console

# ASCII-only markers for Windows cp1252 terminals (Unicode arrows fail in PowerShell).
console = Console(force_terminal=True, legacy_windows=sys.platform == "win32")


def step_start(name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    console.print(f"[bold cyan]>> {name}[/bold cyan]{suffix}")


def step_done(name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    console.print(f"[green]OK {name}[/green]{suffix}")


def paper_step(agent: str, index: int, total: int, paper_id: str, action: str) -> None:
    console.print(f"  [dim]({index}/{total})[/dim] [yellow]{agent}[/yellow] {paper_id}: {action}")


def info(message: str) -> None:
    console.print(f"[dim]  {message}[/dim]")


def warn(message: str) -> None:
    console.print(f"[yellow]  ! {message}[/yellow]")


def route(message: str) -> None:
    console.print(f"[magenta]>> {message}[/magenta]")
