"""
evaluation/eval_utils.py
========================
Shared utilities for all four track evaluators.
- Loads questions
- Calls the LLM to generate an answer from retrieved context
- Saves results to CSV
Scoring is excluded for now.
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[1] / "config" / ".env")

# ── LLM Client ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL"),
    timeout=60,
)
MODEL = os.getenv("SAIA_DEFAULT_MODEL")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Question Loading ───────────────────────────────────────────────────────────

def load_questions(path: str = None) -> list[dict]:
    """Loads questions from questions.json."""
    questions_path = path or Path(__file__).parent / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Answer Generation ──────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """You are a medical assistant for NASA ISS spaceflight emergency procedures.
Answer the question using ONLY the provided context from the procedure documents.
Be concise, accurate, and step-oriented. If the context does not contain enough
information to answer, say: "The provided context does not contain enough information."
"""

def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generates an answer from the retrieved context chunks.
    context_chunks: list of dicts, each must have a 'text' key at minimum.
    """
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
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── CSV Saving ─────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "track", "model", "question_id", "question",
    "sources_retrieved", "answer",
]

def save_results(results: list[dict], track: str, model: str) -> Path:
    """Saves results to a timestamped CSV in evaluation/results/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_").replace("-", "_")
    filename = RESULTS_DIR / f"{track}_{safe_model}_{timestamp}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Results saved to {filename}")
    return filename


def print_summary(results: list[dict], track: str):
    """Prints a simple per-question summary to terminal."""
    if not results:
        print("No results to summarize.")
        return

    avg_s = sum(r["sources_retrieved"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"  {track.upper()} — {len(results)} questions answered")
    print(f"  Avg sources retrieved per question: {avg_s:.1f}")
    print(f"{'='*60}")
    print(f"\n  {'ID':<6}  {'Sources':>7}  Answer preview")
    print(f"  {'-'*56}")
    for r in results:
        preview = r["answer"][:60].replace("\n", " ")
        print(f"  {r['question_id']:<6}  {r['sources_retrieved']:>7}  {preview}...")