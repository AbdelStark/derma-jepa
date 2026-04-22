from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from derma_jepa.observability import (
    log_event,
    progress_iter,
    progress_log_path,
    stage,
)


def _parse_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_event_writes_stdout_and_run_dir(tmp_path: Path) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        log_event("demo.event", run_dir=tmp_path, value=42, label="x")

    stdout_record = json.loads(buffer.getvalue().strip())
    assert stdout_record["event"] == "demo.event"
    assert stdout_record["value"] == 42
    assert stdout_record["label"] == "x"
    assert stdout_record["level"] == "info"

    jsonl_records = _parse_jsonl(progress_log_path(tmp_path))
    assert len(jsonl_records) == 1
    assert jsonl_records[0]["event"] == "demo.event"


def test_stage_emits_start_and_end_with_duration(tmp_path: Path) -> None:
    with redirect_stdout(io.StringIO()), stage(
        "demo.stage", run_dir=tmp_path, extra="yes"
    ) as ctx:
        ctx.set(finished=True)

    jsonl_records = _parse_jsonl(progress_log_path(tmp_path))
    events = [record["event"] for record in jsonl_records]
    assert events == ["demo.stage.start", "demo.stage.end"]
    end_record = jsonl_records[1]
    assert end_record["finished"] is True
    assert isinstance(end_record["duration_seconds"], float)
    assert end_record["duration_seconds"] >= 0.0


def test_stage_records_error_and_reraises(tmp_path: Path) -> None:
    with redirect_stdout(io.StringIO()), pytest.raises(ValueError), stage(
        "demo.stage", run_dir=tmp_path
    ):
        raise ValueError("nope")

    jsonl_records = _parse_jsonl(progress_log_path(tmp_path))
    end_record = jsonl_records[-1]
    assert end_record["event"] == "demo.stage.end"
    assert end_record["level"] == "error"
    assert end_record["error_type"] == "ValueError"
    assert end_record["error_message"] == "nope"


def test_progress_iter_emits_final_tick_with_done_flag(tmp_path: Path) -> None:
    with redirect_stdout(io.StringIO()):
        consumed = list(
            progress_iter(
                range(3),
                name="demo.iter",
                total=3,
                run_dir=tmp_path,
                every=10,
                heartbeat_seconds=0,
            )
        )
    assert consumed == [0, 1, 2]

    jsonl_records = _parse_jsonl(progress_log_path(tmp_path))
    assert jsonl_records[-1]["event"] == "demo.iter.progress"
    assert jsonl_records[-1]["done"] is True
    assert jsonl_records[-1]["processed"] == 3
    assert jsonl_records[-1]["total"] == 3


def test_progress_jsonl_is_append_only(tmp_path: Path) -> None:
    with redirect_stdout(io.StringIO()):
        log_event("a", run_dir=tmp_path)
        log_event("b", run_dir=tmp_path)
        log_event("c", run_dir=tmp_path)

    jsonl_records = _parse_jsonl(progress_log_path(tmp_path))
    assert [record["event"] for record in jsonl_records] == ["a", "b", "c"]


def test_log_event_coerces_non_scalar_fields_without_crashing(tmp_path: Path) -> None:
    class Widget:
        def __repr__(self) -> str:
            return "Widget()"

    with redirect_stdout(io.StringIO()):
        log_event("demo.event", run_dir=tmp_path, widget=Widget())

    payload = _parse_jsonl(progress_log_path(tmp_path))[0]
    assert payload["widget"] == "Widget()"


def test_log_event_survives_unwritable_run_dir(tmp_path: Path) -> None:
    bad_path = tmp_path / "no-write"
    bad_path.mkdir()
    bad_path.chmod(0o500)
    try:
        with redirect_stdout(io.StringIO()) as buffer:
            log_event("demo.event", run_dir=bad_path, value="still logged")
        assert json.loads(buffer.getvalue().strip())["event"] == "demo.event"
    finally:
        bad_path.chmod(0o700)
