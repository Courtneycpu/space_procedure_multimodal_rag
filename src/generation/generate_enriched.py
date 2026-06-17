"""
generate_track2_enriched.py
==================
Track 2 — Enriched Markdown
Retrieves context from the baseline Chroma vector store and generates answers.

Output: data/results/track2_enriched/{model_name}/results.txt

Usage:
    python generate_track2_enriched.py
    python generate_track2_enriched.py --top-k 5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
load_dotenv(ROOT_DIR / "config" / ".env")

from retrieval.search_enriched import retrieve_enriched_context

# ── LLM Client ─────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL"),
    timeout=60,
)
MODEL = os.getenv("SAIA_DEFAULT_MODEL")

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical assistant for NASA ISS spaceflight emergency procedures.
Answer the question using ONLY the provided context from the procedure documents.
Be concise, accurate, and step-oriented.
If the context does not contain enough information to answer, say:
"The provided context does not contain enough information."
If a figure is referenced but not described, say: "see figure <reference>".
"""

# ── Answer Generation ──────────────────────────────────────────────────────────

def generate_answer(question: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return "No context retrieved."

    context_parts = []
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text") or chunk.get("chunk_text") or chunk.get("step_text") or ""
        doc  = chunk.get("doc", "unknown")
        if text:
            context_parts.append(f"[{i+1}] (from {doc})\n{text.strip()}")

    context_str = "\n\n".join(context_parts) if context_parts else "No context available."

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── Questions ──────────────────────────────────────────────────────────────────

def load_questions(path: Path = None) -> list[dict]:
    questions_path = path or Path(__file__).parent / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Output ─────────────────────────────────────────────────────────────────────

def get_output_path() -> Path:
    safe_model = (MODEL or "unknown").replace("/", "_").replace("-", "_")
    out_dir = ROOT_DIR / "data" / "results" / "track2_enriched" / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "results.txt"


def save_results(results: list[dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Track 2 — Enriched Markdown\n")
        f.write(f"Model     : {MODEL}\n")
        f.write(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"[{r['question_id']}] {r['question']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Sources retrieved : {r['sources_retrieved']}\n")
            f.write(f"Answer:\n{r['answer']}\n")
            f.write("\n")

    print(f"\n💾 Results saved to {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(top_k: int):
    questions   = load_questions()
    output_path = get_output_path()
    results     = []

    print(f"\n🚀 Track 2 — Enriched Markdown")
    print(f"   Model     : {MODEL}")
    print(f"   Top-K     : {top_k}")
    print(f"   Questions : {len(questions)}")
    print(f"   Output    : {output_path}\n")

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:70]}...")

        context = retrieve_enriched_context(query=question, top_k=top_k)
        answer  = generate_answer(question=question, context_chunks=context)
        sources = len({c.get("doc") for c in context if c.get("doc")})

        results.append({
            "question_id":       qid,
            "question":          question,
            "sources_retrieved": sources,
            "answer":            answer,
        })
        print(f"     ✓ {sources} source(s) — {answer[:60].replace(chr(10), ' ')}...")

    save_results(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(top_k=args.top_k)