"""
Run all four generation tracks without duplicating their pipeline logic.

Each track keeps its own retrieval, prompt, answer formatting, and result
writer inside its generate_*.py file. This script only orchestrates which
tracks/models/question file to run.

Usage:
    python src/generation/evaluate.py
    python src/generation/evaluate.py --questions src/generation/questions.json
    python src/generation/evaluate.py --top-k 8
    python src/generation/evaluate.py --models gemma-4-31b-it medgemma-27b-it
    python src/generation/evaluate.py --tracks 1 3 4

Output:
    data/results/{question.category}/{question_id}/{model_name}.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
load_dotenv(ROOT_DIR / "config" / ".env")

from generation import generate_enriched
from generation import generate_hybrid
from generation import generate_kg
from generation import generate_text


DEFAULT_MODELS = [
    "medgemma-27b-it",
    "deepseek-r1-distill-llama-70b",
    "qwen3.5-122b-a10b",
    "openai-gpt-oss-120b",
]


TRACKS: dict[int, tuple[str, ModuleType]] = {
    1: ("Pure Text Vector Baseline", generate_text),
    2: ("Enriched Markdown", generate_enriched),
    3: ("Knowledge Graph Retrieval", generate_kg),
    4: ("Hybrid RAG + Knowledge Graph", generate_hybrid),
}


def _safe_model_name(model: str | None) -> str:
    return (model or "unknown").replace("/", "_").replace("-", "_")


def _question_category(questions_path: Path | None) -> str:
    questions_file = questions_path or Path(__file__).parent / "questions.json"
    return questions_file.stem


def _safe_path_part(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip()
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in raw)
    return safe.strip("_") or fallback


def _category_for_question(q: dict, questions_path: Path | None) -> str:
    return _safe_path_part(q.get("category"), fallback=_question_category(questions_path))


def _set_track_model(module: ModuleType, model: str) -> None:
    # The generate_* modules read MODEL when building prompts/output paths.
    # Keeping this assignment here lets evaluate.py reuse their run() methods
    # without reimplementing their internals.
    module.MODEL = model


def _load_questions(questions_path: Path | None) -> list[dict]:
    return generate_text.load_questions(questions_path)


def _save_model_question_file(
    model: str,
    category: str,
    question_result: dict,
    selected_tracks: list[int],
) -> None:
    question_id = question_result["question_id"]
    out_dir = (
        ROOT_DIR
        / "data"
        / "results"
        / category
        / question_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_model_name(model)}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Model       : {model}\n")
        f.write(f"Category    : {category}\n")
        f.write(f"Question ID : {question_id}\n")
        f.write(f"Question    : {question_result['question']}\n")
        f.write(f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for track_num in selected_tracks:
            label = TRACKS[track_num][0]
            result = question_result["tracks"].get(track_num)

            f.write(f"TRACK {track_num}: {label}\n")
            f.write("-" * 60 + "\n")

            if result is None:
                f.write("No result produced.\n\n")
                continue

            f.write(f"Sources retrieved : {result.get('sources_retrieved', 0)}\n")
            f.write("Answer:\n")
            f.write(f"{result.get('answer', '')}\n\n")

    print(f"  Saved {out_path}")


def run(
    top_k: int = 5,
    questions_path: Path | None = None,
    models: list[str] | None = None,
    tracks: list[int] | None = None,
    call_delay: float = 5.0,
) -> None:
    models = models or DEFAULT_MODELS
    selected_tracks = tracks or sorted(TRACKS)
    questions = _load_questions(questions_path)
    categories = sorted({_category_for_question(q, questions_path) for q in questions})

    print("\nEvaluation")
    print(f"  Models   : {len(models)}")
    print(f"  Tracks   : {selected_tracks}")
    print(f"  Categories: {categories}")
    print(f"  Questions: {len(questions)}")
    print(f"  Top-k    : {top_k}")
    print(f"  Call delay: {call_delay}s")
    print(f"  Output   : {ROOT_DIR / 'data' / 'results'}\n")

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model}")
        print(f"{'=' * 60}")

        for q in questions:
            question_id = q["id"]
            category = _category_for_question(q, questions_path)
            print(f"\n[{question_id}] {q['question']}")
            print(f"Category: {category}")

            question_result = {
                "question_id": question_id,
                "question": q["question"],
                "tracks": {},
            }

            for track_num in selected_tracks:
                if track_num not in TRACKS:
                    raise ValueError(f"Unknown track: {track_num}. Expected one of {sorted(TRACKS)}.")

                label, module = TRACKS[track_num]
                _set_track_model(module, model)

                print(f"\nTrack {track_num}: {label}")
                result = module.run_question(q, top_k=top_k)
                question_result["tracks"][track_num] = result
                if call_delay > 0:
                    time.sleep(call_delay)

            _save_model_question_file(
                model=model,
                category=category,
                question_result=question_result,
                selected_tracks=selected_tracks,
            )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run selected generation tracks by calling their generate_*.py run() methods."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks / seed nodes to retrieve per track.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to questions JSON file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override the model list.",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        type=int,
        choices=sorted(TRACKS),
        default=None,
        help="Track numbers to run. Defaults to all tracks.",
    )
    parser.add_argument(
        "--call-delay",
        type=float,
        default=5.0,
        help="Seconds to wait after each track answer call.",
    )
    args = parser.parse_args()

    run(
        top_k=args.top_k,
        questions_path=args.questions,
        models=args.models,
        tracks=args.tracks,
        call_delay=args.call_delay,
    )
