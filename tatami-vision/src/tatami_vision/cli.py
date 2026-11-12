"""CLI `tatami`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from .api import analyze_video, list_backends
from .render import render_video
from .schema import analysis_result_json_schema, label_file_json_schema

app = typer.Typer(help="TatamiLens: análisis de videos de BJJ.", no_args_is_help=True)


@app.command()
def analyze(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    backend: str = typer.Option("ollama", help="ollama | anthropic | local | mock"),
    model: str | None = typer.Option(None, help="Modelo (default según backend)"),
    fps: float = typer.Option(2.0, help="Frames por segundo a muestrear"),
    out: Path | None = typer.Option(None, help="Directorio de salida (default outputs/<video>)"),
    render: bool = typer.Option(False, "--render", help="Genera labeled.mp4 con overlay"),
    crop: bool = typer.Option(False, "--crop/--no-crop", help="Recorta a la zona de combate"),
    smooth: bool = typer.Option(True, "--smooth/--no-smooth"),
    block_size: int | None = typer.Option(None, help="Frames por request al backend"),
    pose_model: str = typer.Option("yolo11m-pose.pt", help="Pesos YOLO-pose para --crop"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Analiza VIDEO y guarda result.json (y opcionalmente labeled.mp4)."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    out = out or Path("outputs") / video.stem
    out.mkdir(parents=True, exist_ok=True)

    def progress(p: float) -> None:
        typer.echo(f"\rProgreso: {p * 100:5.1f}%", nl=False, err=True)

    result = analyze_video(
        video, backend=backend, model=model, sample_fps=fps, progress_cb=progress,
        crop=crop, smooth=smooth, block_size=block_size, pose_model=pose_model,
    )  # fmt: skip
    typer.echo("", err=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    for k in ("A", "B"):
        typer.echo(f"{result['athletes'][k]['label']}: {result['athletes'][k]['description']}")
    typer.echo("\nSegmentos:")
    for s in result["segments"]:
        top = f" (top {s['top']})" if s["top"] != "none" else ""
        mark = "" if s["stable"] else "  ~"
        typer.echo(f"  {s['start']:6.1f}-{s['end']:6.1f}s  {s['position']}{top}{mark}")
    typer.echo("\nEventos:")
    for e in result["events"]:
        typer.echo(f"  {e['t']:6.1f}s  +{e['points']} {e['athlete']}  {e['reason']}")
    sc = result["score"]
    typer.echo(f"\nMarcador: A {sc['A']} - {sc['B']} B\nResultado: {out / 'result.json'}")
    if render:
        path = render_video(result, video, out / "labeled.mp4")
        typer.echo(f"Video: {path}")


@app.command("render")
def render_cmd(
    result: Path = typer.Argument(..., exists=True, dir_okay=False),
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, help="Archivo de salida (default <result>/labeled.mp4)"),
) -> None:
    """Renderiza el video con overlay a partir de RESULT.json."""
    data = json.loads(result.read_text())
    path = render_video(data, video, out or result.with_name("labeled.mp4"))
    typer.echo(f"Video: {path}")


@app.command()
def backends() -> None:
    """Lista backends, modelos y disponibilidad."""
    for b in list_backends():
        flag = "OK " if b["available"] else "NO "
        models = ", ".join(b["models"]) or "-"
        typer.echo(f"[{flag}] {b['name']:10} modelos: {models}")
        if b["detail"]:
            typer.echo(f"       {b['detail']}")


@app.command()
def schema(which: str = typer.Argument("result", help="result | labels")) -> None:
    """Imprime el JSON Schema de AnalysisResult o del formato de etiquetas."""
    sch = analysis_result_json_schema() if which == "result" else label_file_json_schema()
    typer.echo(json.dumps(sch, indent=2))


if __name__ == "__main__":
    app()
