# Pulls unannotated Figure nodes, runs Vision API, updates Neo4j using json
import re
import os
import io
import base64
import json
from pathlib import Path
from PIL import Image
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parents[2]
load_dotenv(ROOT_DIR / "config" / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12344321")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# 1. Initialize Vision LLM Client
client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"), 
    base_url=os.getenv("SAIA_BASE_URL"), 
    timeout=60
)
model_name = os.getenv("SAIA_DEFAULT_MODEL") or "gpt-5.4-mini"


BASE_DIR = BASE_DIR = Path(__file__).parents[2] / "data"

def encode_image(image_path):

    img = Image.open(image_path)
    # Resize if too large
    max_size = (800, 800) # TODO why exactly ?
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def annotate_image(image_path):
    b64 = encode_image(image_path)

    prompt = """You are analyzing a figure from a NASA ISS spaceflight 
    medical procedure document.
    Analyze the image and return a strictly formatted JSON object with exactly these three keys:
    {
      1. "caption": A concise, 1-2 sentence description of what the image shows.
2. "ocr_text": A string containing any visible text, labels, or numbers written inside the image. If none, return "None".
3. "entities": A list of strings representing the physical equipment, body parts, or tools shown in the image.
    }
    Output ONLY valid JSON. Do not use markdown blocks like ```json."""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        temperature=0.1, # TODO what is this for ?
        max_tokens=500
    )

    raw_output = (response.choices[0].message.content or "").strip()
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]
        
    return json.loads(raw_output.strip())

def update_figure_node(path, annotation):
    """Saves the generated annotation back into the Neo4j Knowledge Graph."""
    with driver.session() as session:
        session.run("""
            MATCH (f:Figure {path: $path})
            SET f.caption = $caption,
                f.ocr_text = $ocr_text,
                f.entities = $entities,
                f.annotated = true
        """,
        path=path, 
        caption=annotation.get('caption', ''), 
        ocr_text=annotation.get('ocr_text', ''), 
        entities=annotation.get('entities', []))

# Fetch all unannotated figures from KG
if __name__ == "__main__":
    print("🔍 Scanning Knowledge Graph for unannotated figures...")
    
    # 1. Find all figures that haven't been processed yet
    with driver.session() as session:
        result = session.run("MATCH (f:Figure) WHERE f.annotated = false RETURN f.path AS path")
        figure_paths = [r['path'] for r in result]

    if not figure_paths:
        print("✅ All figures are already annotated in the Knowledge Graph!")
        exit()

    print(f"🚀 Found {len(figure_paths)} figures to annotate.\n")

    #process each figure, annotate, and update KG
    for path in figure_paths:
        fixed_path = re.sub(r'images/\d+_', 'images/', path)

        full_path = os.path.join(BASE_DIR, fixed_path)
        #full_path = os.path.join(BASE_DIR, path)
        print(f"Processing: {path}")

        if not os.path.exists(full_path):
            print(f"  ✗ File not found at {full_path}")
            continue

        try:
            annotation = annotate_image(full_path)
            update_figure_node(path, annotation)
            print(f"  ✅ Caption:  {annotation.get('caption', '')[:60]}...")
            print(f"  ✅ OCR:      {annotation.get('ocr_text', '')[:40]}")
            print(f"  ✅ Entities: {', '.join(annotation.get('entities', []))}\n")
            print()
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")

    print("Annotation complete!")
    driver.close()