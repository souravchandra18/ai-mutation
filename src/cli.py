"""Command-line interface.

Usage:
    python -m src.cli analyze "BRAF V600E"
    python -m src.cli analyze "rs113488022" --json out.json --md out.md
    python -m src.cli evidence "TP53 R175H"
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .evidence import gather
from .llm_config import get_llm_settings
from .reasoning import reason

app = typer.Typer(
    add_completion=False,
    help="Mutation → Mechanism → Therapy reasoning (AMD/OpenAI-compatible LLM).",
)
console = Console()


@app.command()
def evidence(mutation: str = typer.Argument(..., help='e.g. "BRAF V600E" or "rs113488022"')) -> None:
    """Fetch raw evidence only (no LLM call)."""
    mq, ev = gather(mutation)
    console.print(Panel.fit(f"[bold]{mq.label}[/bold]  (raw: {mq.raw})", title="Parsed mutation"))
    console.print_json(data=ev.to_dict())


@app.command()
def analyze(
    mutation: str = typer.Argument(..., help='e.g. "BRAF V600E"'),
    json_out: Path | None = typer.Option(None, "--json", help="Write evidence + reasoning to JSON."),
    md_out: Path | None = typer.Option(None, "--md", help="Write full Markdown report."),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Model id (overrides AI_MODEL / AMD_MODEL env vars).",
    ),
    image_path: Path | None = typer.Option(
        None, "--image",
        help="Optional biomedical image (H&E, radiology, microscopy) scored by BiomedCLIP.",
    ),
    voice_path: Path | None = typer.Option(
        None, "--voice",
        help="Optional voice note (.wav/.mp3/...) transcribed by Whisper.",
    ),
) -> None:
    """Gather evidence then run the 3-stage LLM reasoning chain."""
    image_bytes = image_path.read_bytes() if image_path else None
    voice_bytes = voice_path.read_bytes() if voice_path else None
    with console.status(f"[cyan]Gathering evidence for {mutation}…"):
        mq, ev = gather(mutation, image=image_bytes, voice=voice_bytes)
    console.print(Panel.fit(f"[bold]{mq.label}[/bold]", title="Mutation"))

    settings = get_llm_settings()
    chosen = model or settings.model
    with console.status(
        f"[cyan]Reasoning with {settings.display_provider} "
        f"({chosen} @ {settings.base_url})…"
    ):
        result = reason(mq, ev, model=model)

    md = result.to_markdown(mq.label)
    console.print(Markdown(md))

    if md_out:
        md_out.write_text(md, encoding="utf-8")
        console.print(f"[green]Wrote report →[/green] {md_out}")
    if json_out:
        payload = {
            "mutation": ev.query,
            "evidence": ev.to_dict(),
            "reasoning": {
                "mutation_summary": result.mutation_summary,
                "mechanism": result.mechanism,
                "therapy": result.therapy,
            },
        }
        json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Wrote JSON →[/green] {json_out}")


if __name__ == "__main__":
    app()
