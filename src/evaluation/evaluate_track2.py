import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from retrieval.search_enriched import retrieve_enriched_context
from evaluation.eval_utils import (
    load_questions,
    generate_answer,
    print_summary,
)

TRACK = "track2_enriched"
RESULTS_DIR = Path(__file__).parent / "results"


def save_results_txt(results: list, annotation_model: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = annotation_model.replace("/", "_").replace("-", "_")
    txt_path = RESULTS_DIR / f"{TRACK}_{model_slug}_{timestamp}.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Track:            {TRACK}\n")
        f.write(f"Annotation model: {annotation_model}\n")
        f.write(f"Timestamp:        {timestamp}\n")
        f.write(f"Questions:        {len(results)}\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"[{r['question_id']}] {r['question']}\n")
            f.write(f"Sources retrieved: {r['sources_retrieved']}\n")
            f.write(f"Answer:\n{r['answer']}\n")
            f.write("-" * 60 + "\n\n")

    print(f"💾 Results saved to {txt_path}")
    return txt_path


def run(annotation_model: str, top_k: int):
    questions = load_questions()
    results = []

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
    save_results_txt(results, annotation_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-model", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(annotation_model=args.annotation_model, top_k=args.top_k)