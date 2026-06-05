# Runs Pure KG Structured Search evaluation
"""
Usage:
    python evaluate_track3.py
    python evaluate_track3.py --annotation-model gpt-4o
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from retrieval.search_kg import retrieve_kg_context
from evaluation.eval_utils import (
    load_questions,
    generate_answer,
    save_results,
    print_summary,
    client,
    MODEL,
)

TRACK = "track3_kg"

ENTITY_EXTRACT_PROMPT = """You are a keyword extractor for NASA ISS medical procedures.
Given a question, extract the most important medical or equipment entities
that would appear verbatim in a procedure document.
Return ONLY a valid JSON array of strings, no markdown, no explanation.
Example: ["EpiPen", "CPR", "AED"]

Question: {question}
"""


def extract_entities(question: str) -> list[str]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": ENTITY_EXTRACT_PROMPT.format(question=question)
        }],
        temperature=0.0,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return [e.strip().strip('"') for e in raw.split(",") if e.strip()]


def run(annotation_model: str):
    questions = load_questions()
    results   = []

    print(f"\n🚀 Track 3 — KG Structured Search")
    print(f"   Annotation model : {annotation_model}")
    print(f"   Questions        : {len(questions)}\n")

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:70]}...")

        entities = extract_entities(question)
        print(f"     Entities: {entities}")

        context = retrieve_kg_context(query_entities=entities)
        if not context:
            print(f"     ⚠️  No KG results found")

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
    parser.add_argument("--annotation-model", default="default")
    args = parser.parse_args()
    run(annotation_model=args.annotation_model)