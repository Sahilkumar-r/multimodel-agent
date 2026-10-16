#!/usr/bin/env python
"""CLI for the multimodal RAG agent.

    python main.py                       # ingest only if data changed, then chat
    python main.py --ask "question"      # one-shot answer
    python main.py --reindex             # force a full re-ingest
    python main.py --ingest-only         # sync the knowledge base and exit
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

HELP = """\
[bold]Commands[/bold]
  /upload <path>   add a document or image (pdf, docx, pptx, xlsx, txt, md, csv, html, png, jpg, ...)
  /sources         list ingested files
  /reindex         force a full re-ingest
  /new             start a fresh conversation
  /help            show this help
  /exit            quit
[dim]Videos, audio and URLs / website links cannot be uploaded.[/dim]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal RAG agent (LangGraph + Pinecone + LiteLLM)")
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")
    parser.add_argument("--reindex", action="store_true", help="clear the index and re-ingest everything")
    parser.add_argument("--ingest-only", action="store_true", help="sync the knowledge base and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="show the agent route and guardrail details")
    return parser.parse_args()


def print_report(report) -> None:
    if report is None:
        return
    console.print(
        f"[green]Ingestion finished[/green] ({report.reason}): "
        f"{len(report.added)} added, {len(report.updated)} updated, {len(report.deleted)} deleted, "
        f"{report.vectors} vectors in {report.seconds:.1f}s"
    )
    for name, error in report.failed:
        console.print(f"  [yellow]! {name}: {error}[/yellow]")
    for name, reason in report.skipped:
        console.print(f"  [dim]- skipped {name}: {reason}[/dim]")


def show_answer(result, verbose: bool) -> None:
    console.print(Panel(Markdown(result.answer), title="Assistant", border_style="cyan"))
    if verbose:
        console.print(f"[dim]route: {' > '.join(result.route)}[/dim]")


def main() -> int:
    args = parse_args()

    from src import config  # noqa: F401  (loads .env)

    logging.basicConfig(level=getattr(logging, config.settings.log_level, logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    try:
        with console.status("Loading models and connecting to Pinecone ..."):
            from src.app import RagApp

            app = RagApp(on_event=lambda msg: console.print(f"[dim]{msg}[/dim]"))
    except Exception as exc:  # noqa: BLE001 - configuration problems should be readable
        console.print(f"[red]Startup failed:[/red] {exc}")
        return 1

    if app.tracing:
        console.print("[dim]LangSmith tracing is on.[/dim]")

    report, reason = app.ensure_ingested(force=args.reindex)
    if report is None:
        console.print(f"[green]Knowledge base is up to date, ingestion skipped.[/green] [dim]({app.kb_overview()})[/dim]")
    else:
        print_report(report)
    if args.ingest_only:
        return 0

    thread_id = uuid.uuid4().hex
    if args.ask:
        show_answer(app.ask(args.ask, thread_id), args.verbose)
        return 0

    console.print(Panel(HELP, title="Multimodal RAG Agent", border_style="green"))
    while True:
        try:
            user = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not user:
            continue

        if user.startswith("/"):
            command, _, rest = user.partition(" ")
            command = command.lower()
            if command in ("/exit", "/quit"):
                return 0
            if command == "/help":
                console.print(HELP)
            elif command == "/new":
                thread_id = uuid.uuid4().hex
                console.print("[dim]Started a new conversation.[/dim]")
            elif command == "/sources":
                table = Table("File", "Type", "Vectors", "Note")
                for name, modality, count, note in app.sources():
                    table.add_row(name, modality, str(count), note)
                console.print(table)
            elif command == "/reindex":
                with console.status("Re-indexing ..."):
                    report, _ = app.ensure_ingested(force=True)
                print_report(report)
            elif command == "/upload":
                if not rest.strip():
                    console.print("[yellow]Usage: /upload <path-to-file>[/yellow]")
                    continue
                try:
                    with console.status("Ingesting ..."):
                        report = app.upload(rest)
                    print_report(report) if report else console.print("[dim]Nothing new to ingest.[/dim]")
                except (ValueError, FileNotFoundError) as exc:
                    console.print(f"[yellow]{exc}[/yellow]")
            else:
                console.print("[yellow]Unknown command. Type /help.[/yellow]")
            continue

        try:
            with console.status("Thinking ..."):
                result = app.ask(user, thread_id)
            show_answer(result, args.verbose)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")


if __name__ == "__main__":
    sys.exit(main())
