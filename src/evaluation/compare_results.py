"""
Loads all CSVs from evaluation/results/ and prints a comparison table.
Since there are no scores yet, it compares answer coverage and source counts.

Usage:
    python compare_results.py
    python compare_results.py --model gpt-4o
    python compare_results.py --export summary.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"


def load_all_results(model_filter: str = None) -> list[dict]:
    all_rows = []
    for csv_file in sorted(RESULTS_DIR.glob("*.csv")):
        with open(csv_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if model_filter and row.get("model") != model_filter:
                    continue
                all_rows.append(row)
    return all_rows


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["track"], row["model"])].append(row)

    summary = {}
    for (track, model), group in sorted(groups.items()):
        n        = len(group)
        avg_src  = sum(float(r["sources_retrieved"]) for r in group) / n
        answered = sum(1 for r in group if r["answer"] != "No context retrieved.")
        summary[(track, model)] = {
            "track":        track,
            "model":        model,
            "n_questions":  n,
            "answered":     answered,
            "avg_sources":  round(avg_src, 1),
        }
    return summary


def print_table(summary: dict):
    print(f"\n{'='*65}")
    print(f"  RAG EVALUATION — RESULTS OVERVIEW")
    print(f"{'='*65}")
    print(f"  {'Track':<22} {'Model':<22} {'N':>3}  {'Answered':>8}  {'Avg Src':>7}")
    print(f"  {'-'*60}")
    for v in summary.values():
        print(
            f"  {v['track']:<22} {v['model']:<22} {v['n_questions']:>3}"
            f"  {v['answered']:>8}  {v['avg_sources']:>7.1f}"
        )
    print(f"{'='*65}")
    print(f"  Scoring not yet configured — add it to eval_utils.py when ready.")


def export_summary(summary: dict, path: str):
    fields = ["track", "model", "n_questions", "answered", "avg_sources"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary.values())
    print(f"\n💾 Summary exported to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=None)
    parser.add_argument("--export", default=None)
    args = parser.parse_args()

    rows = load_all_results(model_filter=args.model)
    if not rows:
        print("⚠️  No results found in evaluation/results/")
        print("   Run the track evaluators first.")
        exit(1)

    summary = summarize(rows)
    print_table(summary)

    if args.export:
        export_summary(summary, args.export)