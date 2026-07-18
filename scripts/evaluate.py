#!/usr/bin/env python3
"""Evaluation script using RAGAS to score retrieval and answer quality.

Rewritten for ragas >= 0.4 (modern "collections" API). The legacy functional API
(`Dataset.from_dict`, `ragas.metrics.faithfulness`, `ragas.evaluate(...)`) was
removed/deprecated in this version, so metrics are now instantiated as classes
bound to an LLM + embeddings, and scored directly via `await metric.ascore(...)`.

Usage:
    python -m scripts.evaluate --qa-file data/eval_qa.json

The QA file should be a JSON array of objects with keys:
    "question"        — the test question
    "ground_truth"    — the expected/reference answer

Scores reported (each 0.0-1.0, higher is better):
    - faithfulness      : are all claims in the answer grounded in the retrieved context?
    - answer_relevancy  : does the answer actually address the question?
    - context_precision : are the retrieved chunks relevant, and ranked well?
    - context_recall    : does the retrieved context cover everything in the reference answer?

Notes:
    - RAGAS metrics use an LLM-as-judge. We point that judge at the same local Ollama
      instance the pipeline already uses (via its OpenAI-compatible endpoint), so no
      extra API key is required. Judgment quality depends on the local model, so treat
      absolute scores as directional rather than as precise as a GPT-4-judged run.
    - Answer relevancy and context precision/recall need a `ground_truth` in the QA file;
      faithfulness does not (it only checks the answer against retrieved context).
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402


def _build_judge_llm():
    """LLM-as-judge, backed by the same local Ollama instance used for generation.

    Ollama exposes an OpenAI-compatible endpoint at <host>/v1, so we can reuse
    ragas's standard OpenAI llm_factory path without needing a cloud API key.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    base_url = (settings.ollama_url or "http://localhost:11434").rstrip("/") + "/v1"
    client = AsyncOpenAI(base_url=base_url, api_key="ollama")  # api_key is unchecked by Ollama

    return llm_factory(
        model=settings.ollama_chat_model,
        provider="openai",
        client=client,
        max_tokens=4096,
        temperature=0.0,
    )


def _build_judge_embeddings():
    """Reuse the same HuggingFace embedding model the pipeline uses for retrieval."""
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=settings.hf_embedding_model)


async def _score_one(metrics, question: str, answer: str, contexts: list[str], reference: str) -> dict:
    """Run all four metrics concurrently for a single QA pair."""
    faithfulness, answer_relevancy, context_precision, context_recall = metrics

    results = await asyncio.gather(
        faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        ),
        answer_relevancy.ascore(user_input=question, response=answer),
        context_precision.ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        ),
        context_recall.ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference
        ),
        return_exceptions=True,
    )

    scores = {}
    for name, result in zip(
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        results,
    ):
        if isinstance(result, Exception):
            click.echo(f"    [{name} failed: {result}]")
            scores[name] = None
        else:
            scores[name] = result.value
    return scores


async def _run_evaluation(qa_pairs: list[dict]) -> list[dict]:
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm = _build_judge_llm()
    embeddings = _build_judge_embeddings()

    metrics = (
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),  # with-reference variant
        ContextRecall(llm=llm),
    )

    pipeline = RAGPipeline()
    results = []

    for i, item in enumerate(qa_pairs, start=1):
        question = item["question"]
        reference = item["ground_truth"]
        click.echo(f"\n[{i}/{len(qa_pairs)}] {question}")

        answer_obj = pipeline.query(question)
        pipeline.clear_history()  # each eval question is independent

        answer_1 = answer_obj["answer"]
        answer=re.sub(r"\[\d+(?:\]\[\d+)*\]", "", answer_1)
        contexts = [src["text"] for src in answer_obj["sources"]]

        if not contexts:
            click.echo("    [no retrieved context — skipping scoring]")
            scores = {k: None for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
        else:
            scores = await _score_one(metrics, question, answer, contexts, reference)

        click.echo(
            "    " + ", ".join(f"{k}={v:.3f}" if v is not None else f"{k}=N/A" for k, v in scores.items())
        )

        results.append(
            {
                "question": question,
                "answer": answer,
                "ground_truth": reference,
                **scores,
            }
        )

    return results


def _print_summary(results: list[dict]) -> None:
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    click.echo("\n" + "=" * 50)
    click.echo("EVALUATION RESULTS (mean across questions)")
    click.echo("=" * 50)
    for name in metric_names:
        values = [r[name] for r in results if r.get(name) is not None]
        if values:
            click.echo(f"  {name:<20} {sum(values) / len(values):.4f}  (n={len(values)})")
        else:
            click.echo(f"  {name:<20} N/A (no successful scores)")


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
            '{"question": "...", "ground_truth": "..."} objects. '
            "A starter template has been added for you — edit it with real Q&A "
            "pairs grounded in your data/*.pdf documents before trusting the scores."
        )
        raise SystemExit(1)

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    click.echo(f"Loaded {len(qa_pairs)} QA pairs from {qa_path}")

    results = asyncio.run(_run_evaluation(qa_pairs))

    _print_summary(results)

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"\nPer-question results saved to {output}")


if __name__ == "__main__":
    main()
