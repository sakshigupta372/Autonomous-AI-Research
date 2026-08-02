"""Subprocess sandbox for running generated experiment scripts (Phase 3).

Uses an isolated temp directory and a hard timeout instead of Docker so the
project stays runnable on Windows without extra infrastructure.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    success: bool


def run_python_script(code: str, timeout: int = 30, work_dir: Path | None = None) -> SandboxResult:
    """Execute Python code in a subprocess with timeout and capture output."""
    base_dir = work_dir or Path(tempfile.mkdtemp(prefix="research_sandbox_"))
    base_dir.mkdir(parents=True, exist_ok=True)
    script_path = base_dir / "experiment.py"
    script_path.write_text(code, encoding="utf-8")

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(base_dir),
            check=False,
        )
        return SandboxResult(
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
            exit_code=completed.returncode,
            success=completed.returncode == 0,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return SandboxResult(
            stdout=stdout.strip(),
            stderr=(stderr or f"Execution timed out after {timeout}s").strip(),
            exit_code=-1,
            success=False,
        )
