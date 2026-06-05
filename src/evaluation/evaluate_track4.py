#Hybrid Graph + Vector Search
"""
Usage:
    python evaluate_track4.py --annotation-model gpt-4o
    python evaluate_track4.py --annotation-model claude-3-5-sonnet --top-k 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from retrieval.search_hybrid import retrieve_hybrid_context
from evaluation.eval_utils import (
    load_questions,
    generate_answer,
    save_results,
    print_summary,
)

TRACK = "track4_hybrid"


def format_hybrid_chunks(raw_results: list[dict]) -> list[dict]:
    """
    Merges the richer hybrid retriever output (step_text, captions, ocr)
    into the unified 'text' format that generate_answer() expects.
    """
    formatted = []
    seen = set()

    for r in raw_results:
        parts = []
        chunk_text = r.get("chunk_text") or ""
        step_text  = r.get("step_text")  or ""
        caption    = r.get("llm_caption") or ""
        ocr        = r.get("ocr_text")   or ""

        if chunk_text:
            parts.append(chunk_text)
        if step_text and step_text not in chunk_text:
            parts.append(f"Step: {step_text}")
        if caption:
            parts.append(f"Figure caption: {caption}")
        if ocr and ocr.lower() != "none":
            parts.append(f"Figure OCR: {ocr}")

        combined = " | ".join(parts).strip()
        if not combined or combined in seen:
            continue

        seen.add(combined)
        formatted.append({
            "text":        combined,
            "doc":         r.get("doc", "unknown"),
            "figure_path": r.get("figure_path", ""),
        })

    return formatted


def run(annotation_model: str, top_k: int):
    questions = load_questions()
    results   = []

    print(f"\n🚀 Track 4 — Hybrid Graph + Vector")
    print(f"   Annotation model : {annotation_model}")
    print(f"   Top-K            : {top_k}")
    print(f"   Questions        : {len(questions)}\n")

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:70]}...")

        raw_context = retrieve_hybrid_context(query=question, top_k=top_k)
        context     = format_hybrid_chunks(raw_context)
        figures     = sum(1 for c in context if c.get("figure_path"))
        print(f"     {len(context)} context blocks ({figures} with figures)")

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
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run(annotation_model=args.annotation_model, top_k=args.top_k)