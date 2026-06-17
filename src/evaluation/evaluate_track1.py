# Runs Pure Text Vector Baseline evaluation
"""
Usage:
    python evaluate_track1.py
    python evaluate_track1.py --model gpt-4o --top-k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from retrieval.search_text import retrieve_text_context
from evaluation.eval_utils import (
    load_questions,
    generate_answer,
    save_results,
    print_summary,
)

TRACK = "track1_text"



def run(model: str, top_k: int):
    questions = load_questions()
    results   = []

    print(f"\n🚀 Track 1 — Pure Text Baseline")
    print(f"   Model     : {model}")
    print(f"   Top-K     : {top_k}")
    print(f"   Questions : {len(questions)}\n")

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:70]}...")

        context = retrieve_text_context(query=question, top_k=top_k)
        answer  = generate_answer(question=question, context_chunks=context)
        sources = len({c.get("doc") for c in context if c.get("doc")})

        results.append({
            "track":             TRACK,
            "model":             model,
            "question_id":       qid,
            "question":          question,
            "sources_retrieved": sources,
            "answer":            answer,
        })
        print(f"     ✓ {sources} source(s) — {answer[:60]}...")

    print_summary(results, TRACK)
    save_results(results, TRACK, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="default")
    parser.add_argument("--top-k",  type=int, default=5)
    args = parser.parse_args()
    run(model=args.model, top_k=args.top_k)