"""Lightweight, HF-Jobs-friendly observability for the DermaJEPA pipeline.

Design goals (narrow on purpose):

- Every long-running stage prints one structured line to stdout on start and
  on end, including a duration in seconds. HF Jobs captures stdout directly,
  so this turns silent stretches into legible progress.
- Every stage also appends one JSON-Lines record to
  ``<run_dir>/logs/progress.jsonl`` when a run directory is known. That file
  becomes a cheap, persistent execution trace for the research playbook and
  for future writeups.
- Iterables of non-trivial size get a throttled progress ticker that does not
  depend on a TTY — HF Jobs logs would otherwise swallow a TTY progress bar.
- Zero new runtime dependencies. We do not pull in MLflow / W&B / OpenTelemetry
  for a single-machine research pipeline; those remain available as later
  layers if the project grows into multi-run sweeps.

What we do not do here:

- We do not reimplement metric tracking — run artefacts (``metrics.json``,
  reports under ``artifacts/reports/``) remain the canonical experiment record.
- We do not parse stdout back into structured data — ``progress.jsonl`` is the
  machine-readable mirror.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

_PROGRESS_LOG_NAME = "progress.jsonl"


@dataclass
class StageContext:
    """Mutable handle returned by ``stage()`` so callers can attach fields."""

    name: str
    started_at: float
    run_dir: Path | None
    fields: dict[str, Any] = field(default_factory=dict)

    def set(self, **fields: Any) -> None:
        """Attach fields to the end-of-stage record."""
        self.fields.update(fields)


def log_event(
    name: str,
    *,
    run_dir: Path | None = None,
    level: str = "info",
    **fields: Any,
) -> None:
    """Emit a single structured event to stdout and to the run progress log.

    ``fields`` are serializable values attached to the event. Non-serializable
    values are coerced via ``str()`` so accidental objects never crash the
    pipeline just to log them.
    """
    payload = {
        "ts": _now_iso(),
        "event": name,
        "level": level,
        **_coerce(fields),
    }
    _emit_stdout(payload)
    _emit_jsonl(run_dir, payload)


@contextmanager
def stage(
    name: str,
    *,
    run_dir: Path | None = None,
    **fields: Any,
) -> Iterator[StageContext]:
    """Log a ``stage.start`` / ``stage.end`` pair with duration seconds.

    Wrap the expensive body with::

        with stage("embeddings.export", run_dir=run_dir, model_count=2) as s:
            ...
            s.set(images=10015)

    On exception the end event is emitted with ``level="error"`` and the
    exception class name so a post-mortem can tell what blew up without
    needing to re-run.
    """
    started = time.perf_counter()
    ctx = StageContext(
        name=name,
        started_at=started,
        run_dir=run_dir,
        fields=dict(fields),
    )
    log_event(f"{name}.start", run_dir=run_dir, **ctx.fields)
    try:
        yield ctx
    except BaseException as exc:  # noqa: BLE001 — we re-raise below
        duration = time.perf_counter() - started
        log_event(
            f"{name}.end",
            run_dir=run_dir,
            level="error",
            duration_seconds=round(duration, 3),
            error_type=type(exc).__name__,
            error_message=str(exc)[:400],
            **ctx.fields,
        )
        raise
    else:
        duration = time.perf_counter() - started
        log_event(
            f"{name}.end",
            run_dir=run_dir,
            duration_seconds=round(duration, 3),
            **ctx.fields,
        )


def progress_iter(
    iterable: Iterable[T],
    *,
    name: str,
    total: int | None = None,
    run_dir: Path | None = None,
    every: int = 1,
    heartbeat_seconds: float = 30.0,
) -> Iterator[T]:
    """Yield from ``iterable`` while printing throttled progress ticks.

    Ticks fire whichever comes first: every ``every`` items, or every
    ``heartbeat_seconds``. A final tick fires on clean exhaustion. All ticks
    are flushed to stdout so HF Jobs logs see them in real time.
    """
    processed = 0
    last_tick_time = time.perf_counter()
    start = last_tick_time
    for item in iterable:
        yield item
        processed += 1
        now = time.perf_counter()
        by_count = every > 0 and processed % every == 0
        by_time = heartbeat_seconds > 0 and (now - last_tick_time) >= heartbeat_seconds
        if by_count or by_time:
            log_event(
                f"{name}.progress",
                run_dir=run_dir,
                processed=processed,
                total=total,
                elapsed_seconds=round(now - start, 3),
            )
            last_tick_time = now
    log_event(
        f"{name}.progress",
        run_dir=run_dir,
        processed=processed,
        total=total if total is not None else processed,
        elapsed_seconds=round(time.perf_counter() - start, 3),
        done=True,
    )


def _now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _emit_stdout(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def _emit_jsonl(run_dir: Path | None, payload: dict[str, Any]) -> None:
    if run_dir is None:
        return
    try:
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / _PROGRESS_LOG_NAME
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        # Observability must never crash the run. Swallow filesystem errors
        # (e.g. read-only mounts) and keep stdout as the fallback channel.
        pass


def _coerce(fields: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [_scalar(item) for item in value]
        elif isinstance(value, dict):
            result[key] = {str(k): _scalar(v) for k, v in value.items()}
        else:
            result[key] = str(value)
    return result


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def progress_log_path(run_dir: Path) -> Path:
    """Return the path where progress JSONL entries accumulate for this run."""
    return run_dir / "logs" / _PROGRESS_LOG_NAME


def is_enabled() -> bool:
    """Hook for future runtime disabling; honour DERMA_JEPA_OBS_DISABLE=1."""
    return os.environ.get("DERMA_JEPA_OBS_DISABLE", "").strip() not in {"1", "true"}
