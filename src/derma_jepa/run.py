from __future__ import annotations

import fcntl
import json
import platform
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from derma_jepa.config import PipelineConfig, write_resolved_config


def prepare_run_dir(config: PipelineConfig) -> Path:
    run_dir = config.run_dir
    for relative in (
        "artifacts/embeddings",
        "artifacts/models",
        "artifacts/plots",
        "artifacts/reports",
        "artifacts/demo_cases",
        "logs",
        "fixture/images",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir / "config.yaml")
    write_environment(run_dir / "environment.txt")
    return run_dir


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".derma_jepa.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object: {path}"
        raise ValueError(msg)
    return payload


def append_log(run_dir: Path, name: str, message: str) -> None:
    path = run_dir / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message.rstrip()}\n")


def write_environment(path: Path) -> None:
    lines = [
        f"created_at={datetime.now(UTC).isoformat()}",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"git_commit={_git_commit()}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_run_file(run_dir: Path, relative: str) -> Path:
    path = run_dir / relative
    if not path.exists():
        msg = f"Missing required run artifact: {path}"
        raise FileNotFoundError(msg)
    return path


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()
