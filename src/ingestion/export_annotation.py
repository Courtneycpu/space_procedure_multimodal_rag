# Generates local summary text reports of your visual data, will use it to flattern urls in the raw markdowns

import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# 1. Standardized DB Connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
)

# 2. Relative Pathing
OUTPUT_DIR = os.path.abspath("data/annotations/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_data():
    with driver.session() as session:
        result = session.run("""
            MATCH (f:Figure)
            OPTIONAL MATCH (s:Step)-[:HAS_FIGURE]->(f)
            RETURN f.path AS path,
                   f.caption_text AS original_caption,
                   f.caption AS caption,
                   f.ocr_text AS ocr_text,
                   f.entities AS entities,
                   f.annotated AS annotated,
                   s.text AS step_text,
                   s.number AS step_number
            ORDER BY f.path
        """)
        figures = [dict(r) for r in result]

    print(f"Exporting annotations for {len(figures)} figures...\n")

    json_export_data = []

    # 3. Export human-readable .txt reports
    for fig in figures:
        clean_name = fig['path'].replace('images/', '').replace('/', '_').replace('.png', '')
        output_path = os.path.join(OUTPUT_DIR, f"{clean_name}.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"FIGURE ANNOTATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"File:             {fig['path']}\n")
            f.write(f"Annotated:        {fig['annotated']}\n\n")

            f.write("-" * 60 + "\n")
            f.write("LINKED PROCEDURE STEP:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Step {fig['step_number']}: {fig['step_text']}\n\n")

            f.write("-" * 60 + "\n")
            f.write("LLM GENERATED CAPTION:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{fig['caption'] or 'Not yet annotated'}\n\n")

            f.write("-" * 60 + "\n")
            f.write("OCR TEXT (visible text in image):\n")
            f.write("-" * 60 + "\n")
            f.write(f"{fig['ocr_text'] or 'None'}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("IDENTIFIED ENTITIES:\n")
            f.write("-" * 60 + "\n")
            entities_str = ", ".join(fig['entities']) if fig['entities'] else "None"
            f.write(f"{entities_str}\n\n")

        # Prepare data for JSON export
        json_export_data.append({
            "figure_path": fig["path"],
            "caption": fig["caption"],
            "ocr_text": fig["ocr_text"],
            "entities": fig["entities"]
        })

    # 4. Export the JSON file for Track 2 (enrich_markdown.py)
    json_path = os.path.join(OUTPUT_DIR, "image_annotations.json")
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_export_data, json_file, indent=4)

    print(f"✅ Exported {len(figures)} text reports to {OUTPUT_DIR}")
    print(f"✅ Exported master JSON file to {json_path}")

if __name__ == "__main__":
    export_data()
    driver.close()