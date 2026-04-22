from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from derma_jepa.baselines import evaluate_baselines
from derma_jepa.benchmark import validate_fixture_run
from derma_jepa.config import load_config
from derma_jepa.demo_export import export_demo_bundle, validate_demo_artifact
from derma_jepa.embeddings import export_fixture_embeddings
from derma_jepa.fixtures import audit_fixture_data, build_fixture_manifest
from derma_jepa.pipeline import run_fixture_pipeline
from derma_jepa.public_data import audit_public_dataset, build_public_manifest

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
manifest_app = typer.Typer(no_args_is_help=True)
baseline_app = typer.Typer(no_args_is_help=True)
demo_app = typer.Typer(no_args_is_help=True)
fixture_app = typer.Typer(no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(manifest_app, name="manifest")
app.add_typer(baseline_app, name="baseline")
app.add_typer(demo_app, name="demo")
app.add_typer(fixture_app, name="fixture")

ConfigOption = Annotated[Path, typer.Option(exists=True, readable=True)]
RunDirOption = Annotated[
    Path, typer.Option(exists=True, file_okay=False, dir_okay=True)
]
OutOption = Annotated[Path, typer.Option()]
SplitOption = Annotated[str, typer.Option()]
AllowNegativeOption = Annotated[bool, typer.Option()]
DryRunOption = Annotated[bool, typer.Option()]
ArtifactOption = Annotated[
    Path | None, typer.Option(exists=True, file_okay=False, dir_okay=True)
]


@data_app.command("audit")
def data_audit(config: ConfigOption) -> None:
    """Audit the configured dataset and write leakage notes."""
    parsed = load_config(config)
    output = (
        audit_public_dataset(parsed)
        if parsed.dataset is not None
        else audit_fixture_data(parsed)
    )
    typer.echo(f"data audit written: {output}")


@manifest_app.command("build")
def manifest_build(config: ConfigOption) -> None:
    """Build and validate the configured pair/window manifest."""
    parsed = load_config(config)
    output = (
        build_public_manifest(parsed)
        if parsed.dataset is not None
        else build_fixture_manifest(parsed)
    )
    typer.echo(f"manifest written: {output}")


@app.command("embed")
def embed(config: ConfigOption) -> None:
    """Export deterministic fixture embeddings for manifest images."""
    output = export_fixture_embeddings(load_config(config))
    typer.echo(f"embeddings written: {output}")


@baseline_app.command("eval")
def baseline_eval(
    config: ConfigOption,
    split: SplitOption = "test",
) -> None:
    """Evaluate cheap baselines and write metric artifacts."""
    output = evaluate_baselines(load_config(config), split=split)
    typer.echo(f"baseline metrics written: {output}")


@app.command("eval")
def eval_command(
    config: ConfigOption,
    split: SplitOption = "test",
) -> None:
    """Evaluate the current fixture-tier scoring path."""
    output = evaluate_baselines(load_config(config), split=split)
    typer.echo(f"metrics written: {output}")


@app.command("benchmark")
def benchmark(
    run: RunDirOption,
    allow_negative_result: AllowNegativeOption = False,
) -> None:
    """Validate a run directory against the fixture acceptance gate."""
    output = validate_fixture_run(run, allow_negative_result=allow_negative_result)
    typer.echo(f"benchmark report written: {output}")


@demo_app.command("export")
def demo_export(
    run: RunDirOption,
    out: OutOption,
) -> None:
    """Export deterministic demo artifacts from a completed run."""
    output = export_demo_bundle(run, out)
    typer.echo(f"demo bundle written: {output}")


@demo_app.callback(invoke_without_command=True)
def demo(
    ctx: typer.Context,
    artifact: ArtifactOption = None,
) -> None:
    """Validate a local exported demo artifact and print the HTML entrypoint."""
    if ctx.invoked_subcommand is not None:
        return
    if artifact is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)
    output = validate_demo_artifact(artifact)
    typer.echo(f"demo html ready: {output}")


@fixture_app.command("pipeline")
def fixture_pipeline(config: ConfigOption) -> None:
    """Run the fixture contract from manifest build through demo export."""
    output = run_fixture_pipeline(load_config(config))
    typer.echo(f"fixture pipeline complete: {output}")


@app.command("train")
def train(
    config: ConfigOption,
    dry_run: DryRunOption = False,
) -> None:
    """Guard the locked training command until the JEPA milestone starts."""
    _ = load_config(config)
    message = (
        "train is locked but intentionally gated for Milestone 3; "
        "Milestone 1 validates manifests, baselines, metrics, and demo artifacts"
    )
    if dry_run:
        typer.echo(message)
        return
    raise typer.BadParameter(message)
