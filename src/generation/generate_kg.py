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
import time
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
LLM_RETRY_DELAYS = [15, 30, 60]

# ── Answer Generation ──────────────────────────────────────────────────────────

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
The retrieved context comes from a Knowledge Graph retrieved by entity matching.
Each result is a JSON object that may contain:
- step_text: the exact procedure step matched to your question
- step_body: surrounding narrative text for context
- linked_figure_description: textual figure caption, OCR text, and extracted figure entities
- warning: safety warning linked to that step
- previous_step / next_step: adjacent steps for sequential context

Priority order: step_text > linked_figure_description > step_body > previous/next steps.
Results may not appear in document order; use step numbers to infer sequence.
Merge duplicate or overlapping step information.
Convert figure annotation content into plain text statements in your answer.
"""

TRACK_NAME = "track_3_kg"
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

def _is_retryable_llm_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "429" in message
        or "rate limit" in message
        or "timeout" in message
        or "temporarily" in message
    )


def create_chat_completion_with_retries(messages: list[dict[str, str]]):
    for attempt in range(len(LLM_RETRY_DELAYS) + 1):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=1300,
            )
        except Exception as e:
            if attempt >= len(LLM_RETRY_DELAYS) or not _is_retryable_llm_error(e):
                raise

            delay = LLM_RETRY_DELAYS[attempt]
            print(f"  LLM call retrying in {delay}s: {e}")
            time.sleep(delay)


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
        if r.get("previous_step_text"):
            entry["previous_step"] = {
                "number": r.get("previous_step_number"),
                "text":   r["previous_step_text"],
            }
        if r.get("next_step_text"):
            entry["next_step"] = {
                "number": r.get("next_step_number"),
                "text":   r["next_step_text"],
            }
        context_json["results"].append(entry)

    try: 
        context_text = json.dumps(context_json, indent=2, default=str)
        response = create_chat_completion_with_retries(
            messages=build_messages(context=context_text, question=question),
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
