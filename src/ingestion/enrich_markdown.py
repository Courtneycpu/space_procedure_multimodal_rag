"""
enrich_markdown.py
==================
Reads a plain-text annotations file produced by annotate_context.py,
then walks raw_markdown/ and replaces every image link with its description.
Enriched files are saved to data/enriched_markdown/ preserving the same
subfolder structure as raw_markdown/.

Annotation file format (one block per image):
    ============================================================
    Image: 1.101_SEVERE_ALLERGIC_REACTION/fig1.png
    ------------------------------------------------------------
    <plain text description>

Usage:
    python enrich_markdown.py
    python enrich_markdown.py --annotations data/results/annotations/my_model/annotations.txt
"""

import re
import argparse
from pathlib import Path


# ── Defaults ───────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parents[2]
RAW_MD_DIR    = ROOT_DIR / "data" / "raw_markdown"
ENRICHED_DIR  = ROOT_DIR / "data" / "enriched_markdown"

# Default: pick the first annotations file found under data/results/annotations/
def find_default_annotations() -> Path | None:
    candidates = sorted(
        (ROOT_DIR / "data" / "results" / "annotations").rglob("annotations.txt")
    )
    return candidates[0] if candidates else None


# ── Parse annotations.txt ──────────────────────────────────────────────────────

def load_annotations(annotations_path: Path) -> dict[str, str]:
    """
    Returns a dict mapping relative image path -> plain-text description.
    e.g. {"1.101_SEVERE_ALLERGIC_REACTION/fig1.png": "This figure shows ..."}
    """
    text = annotations_path.read_text(encoding="utf-8", errors="ignore")
    lookup = {}

    # Each block is separated by the === line
    blocks = re.split(r"={60}\n", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # First line: "Image: <relative_path>"
        lines = block.splitlines()
        if not lines[0].startswith("Image:"):
            continue

        image_rel = lines[0].removeprefix("Image:").strip()
        image_rel = image_rel.replace("\\", "/")   # normalize Windows backslashes  

        # Description is everything after the --- separator line
        sep_idx = next(
            (i for i, l in enumerate(lines) if l.startswith("-" * 10)), None
        )
        if sep_idx is None:
            continue

        description = "\n".join(lines[sep_idx + 1:]).strip()
        if image_rel and description:
            lookup[image_rel] = description

    return lookup


# ── Enrich one markdown file ───────────────────────────────────────────────────

# Matches standard markdown image syntax: ![alt text](path/to/fig.png)
IMAGE_LINK_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

def enrich_content(content: str, folder_name: str, lookup: dict[str, str]) -> str:
    """
    Replaces each markdown image link with the original link followed by
    the plain-text description block, using the folder name to build the
    lookup key (e.g. '1.101_SEVERE_ALLERGIC_REACTION/fig1.png').
    """
    def replace_match(m: re.Match) -> str:
        alt  = m.group(1)
        path = m.group(2)                      # e.g. "fig1.png" or "images/fig1.png"
        filename = Path(path).name             # always just the filename

        # Build the lookup key the same way annotate_context.py writes it
        rel_key = f"{folder_name}/{filename}".replace("\\", "/")

        description = lookup.get(rel_key)
        if description:
            figure_label = alt.strip() or filename
            return (
                f"[{figure_label}]\n\n"
                f"> **[Figure description]** {description}\n"
            )
        # No annotation found — leave the link unchanged
        return m.group(0)

    return IMAGE_LINK_RE.sub(replace_match, content)


# ── Walk raw_markdown/ ─────────────────────────────────────────────────────────

def enrich_all(annotations_path: Path):
    print(f"📄 Annotations : {annotations_path}")
    print(f"📁 Raw markdown : {RAW_MD_DIR}")
    print(f"📁 Output       : {ENRICHED_DIR}\n")

    lookup = load_annotations(annotations_path)
    if not lookup:
        print("❌ No annotations loaded — check the file format.")
        return

    print(f"✅ Loaded {len(lookup)} image annotations.\n")

    md_files = list(RAW_MD_DIR.rglob("*.md"))
    if not md_files:
        print("❌ No markdown files found in raw_markdown/")
        return

    enriched, skipped = 0, 0

    for md_file in sorted(md_files):
        # Preserve subfolder structure relative to RAW_MD_DIR
        rel_path    = md_file.relative_to(RAW_MD_DIR)   # e.g. subdir/1.101_....md or 1.101_....md
        out_path    = ENRICHED_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # The folder_name used during annotation is the markdown stem
        # (matches the image subfolder name, e.g. "1.101_SEVERE_ALLERGIC_REACTION")
        folder_name = md_file.stem

        content     = md_file.read_text(encoding="utf-8", errors="ignore")
        enriched_content = enrich_content(content, folder_name, lookup)

        injected = enriched_content.count("[Figure description]")
        out_path.write_text(enriched_content, encoding="utf-8")

        status = f"  💉 {injected} figure(s) injected" if injected else "  — no figures matched"
        print(f"{rel_path}  {status}")

        if injected:
            enriched += 1
        else:
            skipped += 1

    print(f"\nDone! 📝 {enriched} files enriched, {skipped} files unchanged.")
    print(f"Output: {ENRICHED_DIR}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        default=None,
        help="Path to annotations.txt (default: auto-detect under data/results/annotations/)"
    )
    args = parser.parse_args()

    ann_path = Path(args.annotations) if args.annotations else find_default_annotations()

    if ann_path is None or not ann_path.exists():
        print("❌ No annotations file found. Run annotate_context.py first, or pass --annotations <path>.")
        exit(1)

    enrich_all(ann_path)
