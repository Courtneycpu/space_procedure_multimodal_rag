"""
generate_track3.py
==================
Track 3 — Knowledge Graph Retrieval
Extracts entities from each question, retrieves matching steps/figures/warnings
from Neo4j, then generates an answer.

Output: data/results/{question_category}/track3_kg/{model_name}/results.txt

Usage:
    python generate_track3.py
    python generate_track3.py --questions visual_questions.json
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

from retrieval.search_kg import retrieve_kg_context

# ── LLM Client ─────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL"),
    timeout=120,
)
MODEL = os.getenv("SAIA_DEFAULT_MODEL")

# ── Answer Generation ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical assistant for NASA ISS spaceflight emergency procedures.
Answer the question using ONLY the provided context.
Be concise, accurate, and step-oriented.

Very important:
- Do NOT answer by saying "see Figure", "refer to Figure", or "as shown in the figure".
- If a step mentions a figure and a figure annotation is available, convert the figure information into plain text.
- Use the figure caption, OCR text, and figure entities as part of the answer.
- If the retrieved context only contains a figure reference but no figure annotation, say that the visual details are not available in the retrieved context.
- Do not invent visual details that are not in the caption, OCR text, or entities.

If the context does not contain enough information, say:
"The provided context does not contain enough information."
"""

def generate_answer(question: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return "No context retrieved."

    context_json = {
        "query": question,
        "retrieved_from": "Knowledge Graph",
        "results": []
    }

    for r in context_chunks:
        entry = {
            "source":      r.get("doc", "unknown"),
            "step_number": r.get("step_number"),
            "step_text":   r.get("step_text", ""),
            "step_body":   r.get("step_body") or "",
        }
        if r.get("figure_path"):
            caption = r.get("llm_caption") or r.get("caption") or r.get("original_caption") or ""
            ocr = r.get("ocr_text") or ""
            entities = r.get("figure_entities") or r.get("entities") or []

            if isinstance(entities, list):
                entities_text = ", ".join(str(e) for e in entities if e)
            else:
                entities_text = str(entities)

            entry["linked_figure_description"] = (
                f"Figure path: {r.get('figure_path')}\n"
                f"Figure description: {caption if caption else 'No figure description available.'}\n"
                f"OCR text in figure: {ocr if ocr else 'None'}\n"
                f"Visual entities: {entities_text if entities_text else 'None'}"
            )
        if r.get("warning"):
            entry["warning"] = r["warning"]
        context_json["results"].append(entry)

    try: 
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{json.dumps(context_json, indent=2)}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ✗ LLM call failed: {e}")
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
    out_dir = ROOT_DIR / "data" / "results" / category / "track3_kg" / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "results.txt"


def save_results(results: list[dict], output_path: Path, category: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Track 3 — Knowledge Graph Retrieval\n")
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

    context = retrieve_kg_context(query=question, top_k=top_k)
    answer  = generate_answer(question=question, context_chunks=context)
    sources = len({r.get("doc") for r in context if r.get("doc")})

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

    print(f"\n🚀 Track 3 — Knowledge Graph Retrieval")
    print(f"   Model     : {MODEL}")
    print(f"   Category  : {category}")
    print(f"   Questions : {len(questions)}")
    print(f"   Output    : {output_path if write_results else 'returned to evaluator'}\n")

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:70]}...")

        context = retrieve_kg_context(query=question, top_k = top_k)
        answer  = generate_answer(question=question, context_chunks=context)
        sources = len({r.get("doc") for r in context if r.get("doc")})

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
