"""
create annotation based on the context of the markdown and the image,
we are using the description to enrich the markdown.

Output: data/results/annotations/{model_name}/annotations.txt
Each entry contains the image path and its plain-text description.
"""

import os
import io
import re
import base64
from pathlib import Path
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[2] / "config" / ".env")

client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL"),
    timeout=120
)
model_name = os.getenv("SAIA_DEFAULT_MODEL")

BASE_DIR      = Path(__file__).parents[2] / "data" / "images"
MARKDOWN_DIR  = Path(__file__).parents[2] / "data" / "raw_markdown"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONTEXT_WINDOW = 500  # characters to grab before/after image reference


# preprocess the image by resizing, converting to JPEG, and encoding as base64
def encode_image(image_path: Path) -> str:
    """Resize, convert to JPEG, and return a base64 string."""
    img = Image.open(image_path)
    img.thumbnail((800, 800), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# get related context from the markdown
def find_markdown_context(image_path: Path) -> str:
    """
    Finds context for an image by matching its parent folder name to the
    corresponding markdown file (e.g. images/1.101_SEVERE_ALLERGIC_REACTION/fig1.png
    -> raw_markdown/1.101_SEVERE_ALLERGIC_REACTION.md).
 
    Returns the full markdown content, cleaned of image/link syntax.
    Falls back to searching all markdown files if no direct match is found.
    """
    if not MARKDOWN_DIR.exists():
        return ""
 
    folder_name = image_path.parent.name  # e.g. 1.101_SEVERE_ALLERGIC_REACTION
 
    # Direct match: folder name == markdown stem
    md_file = MARKDOWN_DIR / f"{folder_name}.md"
    if md_file.exists():
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        content = re.sub(r'!\[.*?\]\(.*?\)', '[image]', content)
        content = re.sub(r'\[.*?\]\(.*?\)', '', content)
        return f"[From document: {md_file.stem}]\n{content.strip()}"
 
    # Fallback: only search markdown files whose stem starts with the same
    # procedure prefix as the image folder (e.g. "1.101"), so fig1.png from
    # folder 1.101 can never accidentally match fig1.png inside 1.102.md.
    image_name = image_path.name
    procedure_prefix = folder_name.split("_")[0]  # e.g. "1.101"
 
    for md_file in MARKDOWN_DIR.rglob("*.md"):
        if not md_file.stem.startswith(procedure_prefix):
            continue
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        pattern = re.compile(rf'[^\n]*{re.escape(image_name)}[^\n]*', re.IGNORECASE)
        match = pattern.search(content)
        if match:
            start = max(0, match.start() - CONTEXT_WINDOW)
            end   = min(len(content), match.end() + CONTEXT_WINDOW)
            surrounding = content[start:end].strip()
            surrounding = re.sub(r'!\[.*?\]\(.*?\)', '[image]', surrounding)
            surrounding = re.sub(r'\[.*?\]\(.*?\)', '', surrounding)
            return f"[From document: {md_file.stem}]\n{surrounding}"
 
    return ""


#create annotation based on the context of the markdown and the image.
def annotate_image(image_path: Path, context: str = "") -> str:
    """Calls the VLM and returns a plain-text description of the image."""
    b64 = encode_image(image_path)

    prompt = f"""You are analyzing a figure from a NASA ISS medical procedure document.

The text below is surrounding document context. Use it to understand the purpose
of the image, resolve ambiguous labels, and infer procedural meaning.

DO NOT quote, summarize, or repeat the surrounding document text.
DO NOT mention information that is not visually supported by the image.

Document Context:
{context}

Generate a retrieval-oriented image description that focuses on:
- what is visible in the image
- visible labels and text
- objects, equipment, body parts, and tools
- relationships between visible components
- the role of the image within the procedure

Return only plain text.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        temperature=0.1,
        max_tokens=500
    )

    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        raise ValueError("Model returned empty response")
    return raw.strip()


#Returns data/results/annotations/{model_name}/annotations.txt
def get_output_path() -> Path:
    safe_model = (model_name or "unknown").replace("/", "_").replace("-", "_")
    out_dir = Path(__file__).parents[2] / "data" / "results" / "annotations" / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "annotations.txt"

#Apends one image entry to the shared annotations file.
def save_annotation(output_path: Path, image_path: Path, description: str):
    rel_path = image_path.relative_to(BASE_DIR)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Image: {rel_path}\n")
        f.write("-" * 60 + "\n")
        f.write(description + "\n")
        f.write("\n")


# Collection 
def collect_images(base_dir: Path) -> list[Path]:
    return [
        p for p in base_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def already_annotated(output_path: Path, image_path: Path) -> bool:
    """Check if this image already has an entry in the output file."""
    if not output_path.exists():
        return False
    rel = str(image_path.relative_to(BASE_DIR))
    return rel in output_path.read_text(encoding="utf-8", errors="ignore")


# Main 
if __name__ == "__main__":
    print(f"Model:        {model_name}")
    print(f"Images dir:   {BASE_DIR.resolve()}")
    print(f"Markdown dir: {MARKDOWN_DIR.resolve()}\n")

    if not BASE_DIR.exists():
        print(f"❌ Images directory not found: {BASE_DIR}")
        exit(1)

    images = collect_images(BASE_DIR)
    if not images:
        print("No supported images found.")
        exit()

    output_path = get_output_path()
    print(f"📄 Output file: {output_path}\n")
    print(f"🚀 Found {len(images)} images to annotate.\n")

    success, failed, skipped = 0, 0, 0

    for image_path in images:
        rel = image_path.relative_to(BASE_DIR)

        if already_annotated(output_path, image_path):
            print(f"⏭️  Skipping (already annotated): {rel}")
            skipped += 1
            continue

        print(f"Processing: {rel}")

        context = find_markdown_context(image_path)
        if context:
            print(f"  📄 Context found ({len(context)} chars)")
        else:
            print(f"  📄 No markdown context found")

        try:
            description = annotate_image(image_path, context)
            save_annotation(output_path, image_path, description)
            print(f"  ✅ {description[:80].replace(chr(10), ' ')}...")
            success += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            failed += 1

    print(f"\nDone! ✅ {success} succeeded  ⏭️ {skipped} skipped  ❌ {failed} failed.")
    print(f"📄 Annotations written to: {output_path}")
