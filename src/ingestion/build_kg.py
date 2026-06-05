# Parses raw .md files into a connected Neo4j knowledge graph.

import os
import re
import importlib
from neo4j import GraphDatabase
from dotenv import load_dotenv

try:
    text_splitter_module = importlib.import_module("langchain_text_splitters")
    RecursiveCharacterTextSplitter = text_splitter_module.RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=50, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text):
            chunks = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for start in range(0, len(text), step):
                chunk = text[start:start + self.chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password123")
)

MARKDOWN_DIR = os.path.expanduser("~/space_procedure_multimodal_rag/data/raw_markdown")

STOPWORDS = {
    "able", "about", "above", "after", "again", "against", "all", "also",
    "and", "any", "are", "assisted", "available", "bag", "been", "being",
    "below", "both", "button", "call", "cap", "card", "cl", "could", "doc",
    "does", "end", "every", "fig", "figure", "for", "from", "give", "gray",
    "has", "have", "having", "if", "into", "its", "later", "may", "med",
    "medical", "mcc", "not", "off", "one", "or", "other", "out", "over",
    "pack", "page", "pages", "paper", "patient", "perform", "procedure",
    "put", "red", "remove", "request", "retrieve", "sodf", "step", "then",
    "the", "this", "through", "to", "using", "with", "yes",
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)

def parse_markdown(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc_name = os.path.basename(filepath).replace('.md', '')
    steps = []
    figures = []
    warnings = []
    metadata = {}
    current_step = None
    warning_buffer = []
    in_warning = False

    # First line is the title
    if lines:
        metadata['title'] = lines[0].strip()

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Detect OBJECTIVE
        if line_stripped.startswith('OBJECTIVE:'):
            metadata['objective'] = line_stripped.replace('OBJECTIVE:', '').strip()

        # Detect major steps (e.g. "1. DEPLOYING AND USING EPINEPHRINE")
        major_step = re.match(r'^(\d+)\.\s+([A-Z][A-Z\s\(\)]+)$', line_stripped)
        if major_step:
            current_step = {
                'id': f"{doc_name}_step_{major_step.group(1)}",
                'number': major_step.group(1),
                'text': major_step.group(2).strip(),
                'type': 'major',
                'doc': doc_name
            }
            steps.append(current_step)

        # Detect sub steps (e.g. "1.1 Remove Epinephrine...")
        sub_step = re.match(r'^(\d+\.\d+)\s+(.+)$', line_stripped)
        if sub_step:
            current_step = {
                'id': f"{doc_name}_step_{sub_step.group(1).replace('.', '_')}",
                'number': sub_step.group(1),
                'text': sub_step.group(2).strip(),
                'type': 'sub',
                'doc': doc_name
            }
            steps.append(current_step)

        # Detect WARNING blocks
        if line_stripped == 'WARNING':
            in_warning = True
            warning_buffer = []
            continue
        if in_warning:
            if line_stripped == '' and warning_buffer:
                warnings.append({
                    'id': f"{doc_name}_warning_{len(warnings)}",
                    'text': ' '.join(warning_buffer),
                    'step_id': current_step['id'] if current_step else None,
                    'doc': doc_name
                })
                in_warning = False
                warning_buffer = []
            else:
                if line_stripped:
                    warning_buffer.append(line_stripped)

        # Detect figures: ![Figure X](path)
        fig_match = re.match(r'!\[(.+?)\]\((.+?)\)', line_stripped)
        if fig_match:
            # Get caption from next line if it starts with "Figure"
            caption = ''
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('Figure'):
                    caption = next_line

            # Clean up the path
            raw_path = fig_match.group(2)
            clean_path = raw_path.replace('../', '')

            figures.append({
                'id': f"{doc_name}_fig_{len(figures)+1}",
                'label': fig_match.group(1),
                'path': clean_path,
                'caption_text': caption,
                'step_id': current_step['id'] if current_step else None,
                'doc': doc_name
            })

    return doc_name, metadata, steps, figures, warnings

def build_graph(doc_name, metadata, steps, figures, warnings):
    with driver.session() as session:

        # Create Document node
        session.run("""
            MERGE (d:Document {name: $name})
            SET d.title = $title,
                d.objective = $objective
        """,
        name=doc_name,
        title=metadata.get('title', ''),
        objective=metadata.get('objective', ''))

        # Create Step nodes
        for step in steps:
            session.run("""
                MERGE (s:Step {id: $id})
                SET s.number = $number,
                    s.text = $text,
                    s.type = $type,
                    s.doc = $doc
                WITH s
                MATCH (d:Document {name: $doc})
                MERGE (d)-[:HAS_STEP]->(s)
            """, **step)

        # Create Figure nodes
        for fig in figures:
            session.run("""
                MERGE (f:Figure {path: $path})
                SET f.id = $id,
                    f.label = $label,
                    f.caption_text = $caption_text,
                    f.doc = $doc,
                    f.annotated = false,
                    f.caption = null,
                    f.ocr_text = null,
                    f.entities = [],
                    f.embedding = null
                WITH f
                MATCH (d:Document {name: $doc})
                MERGE (d)-[:HAS_FIGURE]->(f)
            """,
            id=fig['id'],
            path=fig['path'],
            label=fig['label'],
            caption_text=fig['caption_text'],
            doc=fig['doc'])

            # Link figure to its step
            if fig['step_id']:
                session.run("""
                    MATCH (f:Figure {path: $path})
                    MATCH (s:Step {id: $step_id})
                    MERGE (s)-[:HAS_FIGURE]->(f)
                """,
                path=fig['path'],
                step_id=fig['step_id'])

        # Create Warning nodes
        for warning in warnings:
            session.run("""
                MERGE (w:Warning {id: $id})
                SET w.text = $text,
                    w.doc = $doc
                WITH w
                MATCH (d:Document {name: $doc})
                MERGE (d)-[:HAS_WARNING]->(w)
            """,
            id=warning['id'],
            text=warning['text'],
            doc=warning['doc'])

            if warning['step_id']:
                session.run("""
                    MATCH (w:Warning {id: $id})
                    MATCH (s:Step {id: $step_id})
                    MERGE (s)-[:HAS_WARNING]->(w)
                """,
                id=warning['id'],
                step_id=warning['step_id'])

       

    print(f"  Steps:    {len(steps)}")
    print(f"  Figures:  {len(figures)}")
    print(f"  Warnings: {len(warnings)}")

# Run for all markdown files
print("Building Knowledge Graph...\n")
for filename in sorted(os.listdir(MARKDOWN_DIR)):
    if filename.endswith('.md'):
        filepath = os.path.join(MARKDOWN_DIR, filename)
        print(f"Processing: {filename}")
        doc_name, metadata, steps, figures, warnings = parse_markdown(filepath)
        build_graph(doc_name, metadata, steps, figures, warnings)
        print()

# Add indexes
with driver.session() as session:
    session.run("CREATE INDEX figure_path IF NOT EXISTS FOR (f:Figure) ON (f.path)")
    session.run("CREATE INDEX step_id IF NOT EXISTS FOR (s:Step) ON (s.id)")
    print("Indexes created.")

driver.close()
print("\nKnowledge Graph built successfully!")
