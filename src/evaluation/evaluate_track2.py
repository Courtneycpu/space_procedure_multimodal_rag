# Runs Pure Enriched Text Vector (flattend media) evaluation
"""
Usage:
    python evaluate_track2.py --annotation-model gpt-4o
    python evaluate_track2.py --annotation-model claude-3-5-sonnet --top-k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from retrieval.search_enriched import retrieve_enriched_context
from evaluation.eval_utils import (
    load_questions,
    generate_answer,
    save_results,
    print_summary,
)

TRACK = "track2_enriched"


def run(annotation_model: str, top_k: int):
    questions = load_questions()
    results   = []

    print(f"\n🚀 Track 2 — Enriched Text Vector")
    print(f"   Annotation model : {annotation_model}")
    print(f"   Top-K            : {top_k}")
    print(f"   Questions        : {len(questions)}\n")

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:70]}...")

        context = retrieve_enriched_context(query=question, top_k=top_k)
        answer  = generate_answer(question=question, context_chunks=context)
        sources = len({c.get("doc") for c in context if c.get("doc")})

        results.append({
            "track":             TRACK,
            "model":             annotation_model,
            "question_id":       qid,
            "question":          question,
            "sources_retrieved": sources,
            "answer":            answer,
        })
        print(f"     ✓ {sources} source(s) — {answer[:60]}...")

    print_summary(results, TRACK)
    save_results(results, TRACK, annotation_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-model", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(annotation_model=args.annotation_model, top_k=args.top_k)