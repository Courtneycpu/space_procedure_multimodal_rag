"""
generate_track2_enriched.py
==================
Track 2 — Enriched Markdown
Retrieves context from the baseline Chroma vector store and generates answers.

Output: data/results/{question_category}/track2_enriched/{model_name}/results.txt

Usage:
    python generate_track2_enriched.py
    python generate_track2_enriched.py --top-k 5
    python generate_track2_enriched.py --questions visual_questions.json
"""

import os
import sys
import json
import argparse
import re
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
    timeout=120,
)
MODEL = os.getenv("SAIA_DEFAULT_MODEL")

# ── Prompts ────────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are a procedure-grounded assistant for NASA ISS spaceflight emergency procedure documents.

Your task is to answer the user's question using ONLY the provided retrieved context.

Grounding rules:
1. Use only the provided context. Do not use outside medical or general knowledge.
2. First decide whether the retrieved context contains enough information to answer.
3. If the context is insufficient, answer exactly:
"The provided context does not contain enough information."
4. Ignore irrelevant or duplicate context.
5. If the retrieved context contains conflicting information, say that the retrieved context is conflicting and only give details that are directly supported.
6. Preserve exact procedure names, equipment names, medication names, quantities, units, warnings, and order of steps when they appear in the context.
7. Do not invent missing steps, figure details, warnings, dosages, or equipment.

Textual figure-representation rules:
8. Some pipelines may include textual representations of figures, such as figure descriptions, captions, OCR text, labels, or extracted figure entities.
9. Treat those textual figure representations as evidence.
10. Do not answer only with "see figure", "refer to the figure", or "as shown in the figure" if textual figure evidence is available. Convert figure annotation content into plain text statements in your answer.
11. If a figure is referenced but no textual figure evidence is available, say that the visual details are not available in the retrieved context.
12. Do not include figure-number citations unless the user explicitly asks for them.

Answer style:
13. Be concise, accurate, and step-oriented.
14. For procedure questions, use numbered steps.
15. For equipment or warning questions, use short bullet points.
16. Output only the final answer, not your reasoning process.
"""

TRACK_CONTEXT_NOTE = """Context format:
The retrieved context is enriched markdown converted into plain text.
Lines labeled "Figure description:" are textual evidence describing figure content.
Lines labeled "Image caption:" are figure captions from the source material.
Use figure descriptions and image captions directly when they answer figure-related questions.
Do not mention image paths or markdown image links in the answer.
"""

TRACK_NAME = "track_2_enriched_markdown"
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + TRACK_CONTEXT_NOTE


def build_messages(context: str, question: str) -> list[dict[str, str]]:
    user_prompt = f"""
<retrieval_track>
{TRACK_NAME}
</retrieval_track>

<context_format>
{TRACK_CONTEXT_NOTE.strip()}
</context_format>

<retrieved_context>
{context}
</retrieved_context>

<question>
{question}
</question>
""".strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt},
    ]

# ── Answer Generation ──────────────────────────────────────────────────────────

FIGURE_DESCRIPTION_RE = re.compile(
    r"^\s*>\s*\*\*\[Figure description\]\*\*\s*",
    flags=re.IGNORECASE | re.MULTILINE,
)
IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
FIGURE_REF_RE = re.compile(
    r"\s*\(Figure\s+\d+(?:\s+and\s+Figure\s+\d+)?\)",
    flags=re.IGNORECASE,
)
FIGURE_LABEL_RE = re.compile(r"^\s*\[Figure\s+\d+\]\s*$\n?", flags=re.MULTILINE)
FIGURE_CAPTION_RE = re.compile(
    r"^\s*Figure\s+\d+\.?-?\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)
FIGURE_SENTENCE_RE = re.compile(
    r"\bFigure\s+\d+\s+(shows|illustrates|is|contains|serves)\b",
    flags=re.IGNORECASE,
)


def format_enriched_text(text: str) -> str:
    """Make enriched markdown read like evidence instead of figure pointers."""
    text = IMAGE_LINK_RE.sub(r"[\1 image reference]", text)
    text = FIGURE_DESCRIPTION_RE.sub("Figure description: ", text)
    text = FIGURE_REF_RE.sub("", text)
    text = FIGURE_LABEL_RE.sub("", text)
    text = FIGURE_CAPTION_RE.sub(r"Image caption: \1", text)
    text = FIGURE_SENTENCE_RE.sub(r"The image \1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return "No context retrieved."

    context_parts = []
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text") or chunk.get("chunk_text") or chunk.get("step_text") or ""
        doc  = chunk.get("doc", "unknown")
        if text:
            context_parts.append(f"[{i+1}] (from {doc})\n{format_enriched_text(text)}")

    context_str = "\n\n".join(context_parts) if context_parts else "No context available."


    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(context=context_str, question=question),
            temperature=0.0,
            max_tokens=1300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e: 
        print (f"x LLM call failed: {e}")
        return "Error: LLM call failed."


# ── Questions ──────────────────────────────────────────────────────────────────

def load_questions(path: Path = None) -> list[dict]:
    questions_path = path or Path(__file__).parent / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_questions_category(path: Path = None) -> str:
    questions_path = path or Path(__file__).parent / "questions.json"
    return questions_path.stem


# ── Output ─────────────────────────────────────────────────────────────────────

def get_output_path(category: str) -> Path:
    safe_model = (MODEL or "unknown").replace("/", "_").replace("-", "_")
    out_dir = ROOT_DIR / "data" / "results" / category / "track2_enriched" / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "results.txt"


def save_results(results: list[dict], output_path: Path, category: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Track 2 — Enriched Markdown\n")
        f.write(f"Model     : {MODEL}\n")
        f.write(f"Category  : {category}\n")
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

def run_question(q: dict, top_k: int) -> dict:
    qid      = q["id"]
    question = q["question"]
    print(f"  [{qid}] {question[:70]}...")

    context = retrieve_enriched_context(query=question, top_k=top_k)
    answer  = generate_answer(question=question, context_chunks=context)
    sources = len({c.get("doc") for c in context if c.get("doc")})

    result = {
        "question_id":       qid,
        "question":          question,
        "sources_retrieved": sources,
        "answer":            answer,
    }
    print(f"     âœ“ {sources} source(s) â€” {answer[:60].replace(chr(10), ' ')}...")
    return result


def run(top_k: int, questions_path: Path = None, write_results: bool = True) -> list[dict]:
    questions   = load_questions(questions_path)
    category    = get_questions_category(questions_path)
    output_path = get_output_path(category) if write_results else None
    results     = []

    print(f"\n🚀 Track 2 — Enriched Markdown")
    print(f"   Model     : {MODEL}")
    print(f"   Category  : {category}")
    print(f"   Top-K     : {top_k}")
    print(f"   Questions : {len(questions)}")
    print(f"   Output    : {output_path if write_results else 'returned to evaluator'}\n")

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

    if write_results and output_path is not None:
        save_results(results, output_path, category)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=None)
    args = parser.parse_args()
    run(top_k=args.top_k, questions_path=args.questions)
