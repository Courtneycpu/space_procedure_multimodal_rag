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
 
from retrieval.search_text import retrieve_text_context
 
# ── LLM Client ─────────────────────────────────────────────────────────────────
 
client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL"),
    timeout=60,
)
MODEL = os.getenv("SAIA_DEFAULT_MODEL")

SYSTEM_PROMPT = """You are a medical assistant for NASA ISS spaceflight emergency procedures.
Answer the question using ONLY the provided context from the procedure documents.
Be concise, accurate, and step-oriented.
If the context does not contain enough information to answer, say:
"The provided context does not contain enough information."
If a figure is referenced but not described, say: "see figure <reference>".
"""
 
# Answer Generation
 
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

    try: 
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
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
    out_dir = ROOT_DIR / "data" / "results" / category / "track1" / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "results.txt"
 
 
def save_results(results: list[dict], output_path: Path, category: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Track 1 — Pure Text Vector Baseline\n")
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
 
def run(top_k: int, questions_path: Path = None):
    questions   = load_questions(questions_path)
    category    = get_questions_category(questions_path)
    output_path = get_output_path(category)
    results     = []
 
    print(f"\n🚀 Track 1 — Pure Text Baseline")
    print(f"   Model     : {MODEL}")
    print(f"   Category  : {category}")
    print(f"   Top-K     : {top_k}")
    print(f"   Questions : {len(questions)}")
    print(f"   Output    : {output_path}\n")
 
    for q in questions:
        qid      = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:70]}...")
 
        context = retrieve_text_context(query=question, top_k=top_k)
        answer  = generate_answer(question=question, context_chunks=context)
        sources = len({c.get("doc") for c in context if c.get("doc")})
 
        results.append({
            "question_id":       qid,
            "question":          question,
            "sources_retrieved": sources,
            "answer":            answer,
        })
        print(f"     ✓ {sources} source(s) — {answer[:60].replace(chr(10), ' ')}...")
 
    save_results(results, output_path, category)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=None)
    args = parser.parse_args()
    run(top_k=args.top_k, questions_path=args.questions)
