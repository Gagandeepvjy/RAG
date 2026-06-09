#!/usr/bin/env python3
"""CLI for querying the RAG system."""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.rag import RAGPipeline


@click.command()
@click.argument("question", required=False)
@click.option("--sources", is_flag=True, help="Show retrieved source chunks.")
@click.option("--interactive", "-i", is_flag=True, help="Start an interactive chat session.")
@click.option("--clear-history", is_flag=True, help="Clear conversation history before querying.")
def main(question: str | None, sources: bool, interactive: bool, clear_history: bool) -> None:
    """
    Ask a question against the indexed knowledge base.

    Run with a QUESTION argument for a single query, or use --interactive / -i
    for a multi-turn conversation session.
    """
    pipeline = RAGPipeline()

    if clear_history:
        pipeline.clear_history()
        click.echo("Conversation history cleared.")

    if interactive:
        click.echo("Interactive mode — type 'quit' or 'exit' to stop, 'clear' to reset history.\n")
        while True:
            try:
                q = click.prompt("You", prompt_suffix="> ").strip()
            except (EOFError, KeyboardInterrupt):
                click.echo("\nGoodbye.")
                break

            if q.lower() in ("quit", "exit"):
                click.echo("Goodbye.")
                break
            if q.lower() == "clear":
                pipeline.clear_history()
                click.echo("History cleared.\n")
                continue
            if not q:
                continue

            result = pipeline.query(q)
            _print_result(result, sources)
        return

    if not question:
        click.echo("Provide a QUESTION or use --interactive / -i for chat mode.")
        raise SystemExit(1)

    result = pipeline.query(question)
    _print_result(result, sources)


def _print_result(result: dict, show_sources: bool) -> None:
    click.echo("\n" + "=" * 60)
    click.echo("ANSWER")
    click.echo("=" * 60)
    click.echo(result["answer"])

    if show_sources:
        click.echo("\n" + "=" * 60)
        click.echo("SOURCES")
        click.echo("=" * 60)
        for src in result["sources"]:
            click.echo(f"\n[{src['index']}] {src['source']}")
            click.echo(src["text"])
    click.echo()


if __name__ == "__main__":
    main()
