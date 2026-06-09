#!/usr/bin/env python3
"""Evaluation script using ragas to score retrieval and answer quality.

Usage:
    python scripts/evaluate.py --qa-file data/eval_qa.json

The QA file should be a JSON array of objects with keys:
    "question"        — the test question
    "ground_truth"    — the expected answer (used for faithfulness scoring)

Scores reported:
    - faithfulness      : are all claims in the answer grounded in the context?
    - answer_relevancy  : does the answer address the question?
    - context_precision : are the retrieved chunks relevant?
    - context_recall    : does the retrieved context cover the ground truth?
"""

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.rag import RAGPipeline


@click.command()
@click.option(
    "--qa-file",
    default="data/eval_qa.json",
    show_default=True,
    help="Path to JSON file with question/ground_truth pairs.",
)
@click.option(
    "--output",
    default="eval_results.json",
    show_default=True,
    help="Where to write per-question scores.",
)
def main(qa_file: str, output: str) -> None:
    qa_path = Path(qa_file)
    if not qa_path.exists():
        click.echo(f"QA file not found: {qa_path}")
        click.echo(
            "Create data/eval_qa.json with a list of "
            '{"question": "...", "ground_truth": "..."} objects.'
        )
        raise SystemExit(1)

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    click.echo(f"Loaded {len(qa_pairs)} QA pairs from {qa_path}")

    pipeline = RAGPipeline()

    # Collect ragas inputs
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(qa_pairs, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        click.echo(f"\n[{i}/{len(qa_pairs)}] {q}")

        result = pipeline.query(q)
        pipeline.clear_history()  # each eval question is independent

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([src["text"] for src in result["sources"]])
        ground_truths.append(gt)

    # Run ragas evaluation
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        click.echo("\nRunning ragas evaluation...")
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        click.echo("\n" + "=" * 50)
        click.echo("EVALUATION RESULTS")
        click.echo("=" * 50)
        for metric, score in scores.items():
            click.echo(f"  {metric:<25} {score:.4f}")

        # Save per-question results
        results = scores.to_pandas().to_dict(orient="records")
        for i, row in enumerate(results):
            row["question"] = questions[i]
            row["answer"] = answers[i]
            row["ground_truth"] = ground_truths[i]

        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nPer-question results saved to {output}")

    except ImportError:
        click.echo(
            "\nragas or datasets not installed. "
            "Run: pip install ragas datasets"
        )


if __name__ == "__main__":
    main()
