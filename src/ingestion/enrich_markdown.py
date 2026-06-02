# Generates the flattened text files for Track 2 evaluation

import os
import json
from pathlib import Path

def flatten_markdown():
    raw_dir = Path("data/raw_markdown")
    enriched_dir = Path("data/enriched_markdown")
    enriched_dir.mkdir(parents=True, exist_ok=True)
    
    # Load your annotation cache file
    annotations_path = Path("image_annotations.json")
    annotations = {}
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            data = json.load(f)
            annotations = {item['figure_path']: item for item in data} if isinstance(data, list) else data

    for md_file in raw_dir.glob("*.md"):
        content = md_file.read_text(encoding='utf-8')
        
        # Look for figure inclusions and inject descriptors right below them
        for fig_path, annot in annotations.items():
            fig_filename = os.path.basename(fig_path)
            if fig_filename in content:
                injection = f"\n\n[IMAGE DESCRIPTION: {annot.get('caption')} OCR TEXT found in diagram: {annot.get('ocr_text')}]\n\n"
                content = content.replace(fig_filename, f"{fig_filename}{injection}")
                
        (enriched_dir / md_file.name).write_text(content, encoding='utf-8')
    print("✅ Flattened markdown files created in data/enriched_markdown/")

if __name__ == "__main__":
    flatten_markdown()