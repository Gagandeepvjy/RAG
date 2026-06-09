"""Ingest documents: load -> chunk -> embed -> save index to disk."""

import click
from src.rag import ingest


@click.command()
@click.option("--source", default=None, help="Path to documents directory.")
def main(source):
    ingest(source)


if __name__ == "__main__":
    main()
