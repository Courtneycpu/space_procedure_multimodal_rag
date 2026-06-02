# Parses raw .md files into a connected Neo4j knowledge graph.

import os
import re
import importlib

from dotenv import load_dotenv

GraphDatabase = None
try:
    neo4j_module = importlib.import_module("neo4j")
    GraphDatabase = neo4j_module.GraphDatabase
except ModuleNotFoundError:
    GraphDatabase = None

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

driver = None
if GraphDatabase is not None:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123")),
    )

MARKDOWN_DIR = os.path.abspath("data/raw_markdown/")
MIN_CONCEPT_LEN = 3
MAX_CONCEPTS_PER_NODE = 20

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


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def get_step_level(number):
    return number.count(".") + 1


def extract_figure_number(text):
    match = re.search(r"Figure\s+(\d+)", text or "", re.IGNORECASE)
    return match.group(1) if match else None


def extract_procedure_refs(text):
    refs = set()
    for match in re.finditer(r"\b\d+(?:\.\d+){1,3}\b", text or ""):
        refs.add(match.group(0))
    return sorted(refs)


def extract_concepts(text, limit=MAX_CONCEPTS_PER_NODE):
    """Small deterministic keyword extractor for shared Concept nodes."""
    if not text:
        return []

    candidates = []
    phrase_pattern = (
        r"\b(?:[A-Z][A-Za-z0-9/-]{2,}|[A-Z]{2,})"
        r"(?:\s+(?:[A-Z][A-Za-z0-9/-]{2,}|[A-Z]{2,})){0,3}\b"
    )

    for phrase in re.findall(phrase_pattern, text):
        cleaned = normalize_text(phrase)
        if cleaned.lower() not in STOPWORDS:
            candidates.append(cleaned)

    for word in re.findall(r"[A-Za-z][A-Za-z0-9/-]{2,}", text):
        word_lower = word.lower()
        if word_lower not in STOPWORDS and len(word_lower) >= MIN_CONCEPT_LEN:
            candidates.append(word)

    deduped = []
    seen = set()
    for concept in candidates:
        key = slugify(concept)
        if key and key not in seen:
            deduped.append({"id": key, "name": concept})
            seen.add(key)
        if len(deduped) >= limit:
            break

    return deduped


def detect_step(line):
    line = line.strip()

    if re.match(r"^\d{1,2}\s+[A-Z]{3}\s+\d{2}$", line):
        return None
    if re.match(r"^\d+(?:\.\d+)+_[A-Z0-9_]+\.doc$", line):
        return None

    dotted_major = re.match(r"^(\d+)\.\s+([A-Z][A-Z0-9\s/&(),\-]+)$", line)
    if dotted_major:
        return dotted_major.group(1), normalize_text(dotted_major.group(2)), "major"

    dotted_sub = re.match(r"^(\d+\.\d{1,2}(?:\.\d{1,2})*)\s+(.+)$", line)
    if dotted_sub:
        return dotted_sub.group(1), normalize_text(dotted_sub.group(2)), "sub"

    block_step = re.match(r"^(\d+)\s+([A-Z][A-Z0-9\s/&(),\-]+)$", line)
    if block_step:
        return block_step.group(1), normalize_text(block_step.group(2)), "major"

    return None


def finalize_step_body(steps, lines, end_idx):
    if not steps:
        return

    step = steps[-1]
    if step.get("body") is not None:
        return

    body_lines = []
    for body_line in lines[step["start_line"] + 1:end_idx]:
        body_line = body_line.strip()
        if not body_line or body_line == "WARNING":
            continue
        if body_line.startswith("![") or body_line.startswith("Figure"):
            continue
        body_lines.append(body_line)

    step["body"] = normalize_text(" ".join(body_lines))


def clean_table_cell(cell):
    return normalize_text(cell.replace("<br>", " ").strip())


def parse_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    lines = []
    for raw_line in raw_text.replace("<br>", "\n").splitlines():
        cleaned = raw_line.strip().strip("|").strip()
        if not cleaned or re.fullmatch(r"[-|\s]+", cleaned):
            lines.append("")
            continue
        lines.append(cleaned)

    doc_name = os.path.basename(filepath).replace(".md", "")
    steps = []
    figures = []
    warnings = []
    metadata = {}
    current_step = None
    step_by_number = {}
    warning_buffer = []
    in_warning = False

    if lines:
        metadata["title"] = lines[0].strip()

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if line_stripped == metadata.get("title"):
            continue

        if line_stripped.startswith("OBJECTIVE:"):
            metadata["objective"] = line_stripped.replace("OBJECTIVE:", "").strip()

        detected_step = detect_step(line_stripped)
        if detected_step:
            finalize_step_body(steps, lines, i)
            number, text, step_type = detected_step
            parent_id = None

            if get_step_level(number) > 1:
                parent_number = ".".join(number.split(".")[:-1])
                parent_id = step_by_number.get(parent_number)

            current_step = {
                "id": f"{doc_name}_step_{number.replace('.', '_')}",
                "number": number,
                "text": text,
                "body": None,
                "type": step_type,
                "level": get_step_level(number),
                "parent_id": parent_id,
                "doc": doc_name,
                "start_line": i,
            }
            steps.append(current_step)
            step_by_number[number] = current_step["id"]

        if line_stripped == "WARNING":
            in_warning = True
            warning_buffer = []
            continue

        if in_warning:
            if line_stripped == "" and warning_buffer:
                warnings.append(
                    {
                        "id": f"{doc_name}_warning_{len(warnings)}",
                        "text": normalize_text(" ".join(warning_buffer)),
                        "step_id": current_step["id"] if current_step else None,
                        "doc": doc_name,
                    }
                )
                in_warning = False
                warning_buffer = []
            elif line_stripped:
                warning_buffer.append(line_stripped)

        fig_match = re.match(r"!\[(.+?)\]\((.+?)\)", line_stripped)
        if fig_match:
            caption = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("Figure"):
                    caption = next_line

            clean_path = fig_match.group(2).replace("../", "")

            figures.append(
                {
                    "id": f"{doc_name}_fig_{len(figures) + 1}",
                    "label": fig_match.group(1),
                    "number": extract_figure_number(fig_match.group(1)),
                    "path": clean_path,
                    "caption_text": caption,
                    "step_id": current_step["id"] if current_step else None,
                    "doc": doc_name,
                }
            )

        caption_number = extract_figure_number(line_stripped)
        if (
            caption_number
            and line_stripped.startswith("Figure")
            and not fig_match
            and caption_number not in {fig["number"] for fig in figures}
        ):
            image_path = os.path.join("images", doc_name, f"fig{caption_number}.png")
            if os.path.exists(os.path.join("data", image_path)):
                figures.append(
                    {
                        "id": f"{doc_name}_fig_{len(figures) + 1}",
                        "label": f"Figure {caption_number}",
                        "number": caption_number,
                        "path": image_path.replace("\\", "/"),
                        "caption_text": line_stripped,
                        "step_id": current_step["id"] if current_step else None,
                        "doc": doc_name,
                    }
                )

    finalize_step_body(steps, lines, len(lines))

    if in_warning and warning_buffer:
        warnings.append(
            {
                "id": f"{doc_name}_warning_{len(warnings)}",
                "text": normalize_text(" ".join(warning_buffer)),
                "step_id": current_step["id"] if current_step else None,
                "doc": doc_name,
            }
        )

    if not steps:
        current_observation = ""
        row_number = 1
        for raw_line in raw_text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line.startswith("|") or "---" in raw_line or "Observation" in raw_line:
                continue

            cells = [clean_table_cell(cell) for cell in raw_line.strip("|").split("|")]
            if len(cells) < 3:
                continue

            observation, possible_cause, action = cells[:3]
            if observation:
                current_observation = observation
            if not current_observation or not action:
                continue

            steps.append(
                {
                    "id": f"{doc_name}_table_row_{row_number}",
                    "number": f"T{row_number}",
                    "text": current_observation,
                    "body": normalize_text(f"Cause: {possible_cause}. Action: {action}"),
                    "type": "table_row",
                    "level": 1,
                    "parent_id": None,
                    "doc": doc_name,
                    "start_line": 0,
                }
            )
            row_number += 1

    return doc_name, metadata, steps, figures, warnings


def link_concepts(session, node_label, node_key, node_key_value, text):
    for concept in extract_concepts(text):
        session.run(
            f"""
            MERGE (c:Concept {{id: $concept_id}})
            SET c.name = $concept_name
            WITH c
            MATCH (n:{node_label} {{{node_key}: $node_key_value}})
            MERGE (n)-[:MENTIONS]->(c)
            """,
            concept_id=concept["id"],
            concept_name=concept["name"],
            node_key_value=node_key_value,
        )


def link_procedure_refs(session, node_label, node_key, node_key_value, text):
    for ref in extract_procedure_refs(text):
        session.run(
            f"""
            MERGE (p:ProcedureRef {{code: $code}})
            WITH p
            MATCH (n:{node_label} {{{node_key}: $node_key_value}})
            MERGE (n)-[:REFERENCES_PROCEDURE]->(p)
            """,
            code=ref,
            node_key_value=node_key_value,
        )


def build_graph(doc_name, metadata, steps, figures, warnings):
    if driver is None:
        raise RuntimeError("Install neo4j before building the graph: pip install -r config/requirements.txt")

    with driver.session() as session:
        session.run(
            """
            MERGE (d:Document {name: $name})
            SET d.title = $title,
                d.objective = $objective
            """,
            name=doc_name,
            title=metadata.get("title", ""),
            objective=metadata.get("objective", ""),
        )

        for step in steps:
            session.run(
                """
                MERGE (s:Step {id: $id})
                SET s.number = $number,
                    s.text   = $text,
                    s.body   = $body,
                    s.type   = $type,
                    s.level  = $level,
                    s.doc    = $doc
                WITH s
                MATCH (d:Document {name: $doc})
                MERGE (d)-[:HAS_STEP]->(s)
                """,
                id=step["id"],
                number=step["number"],
                text=step["text"],
                body=step.get("body") or "",
                type=step["type"],
                level=step["level"],
                doc=step["doc"],
            )

            if step.get("parent_id"):
                session.run(
                    """
                    MATCH (parent:Step {id: $parent_id})
                    MATCH (child:Step {id: $child_id})
                    MERGE (parent)-[:HAS_SUBSTEP]->(child)
                    """,
                    parent_id=step["parent_id"],
                    child_id=step["id"],
                )

            step_text = f"{step['text']} {step.get('body') or ''}"
            link_concepts(session, "Step", "id", step["id"], step_text)
            link_procedure_refs(session, "Step", "id", step["id"], step_text)

        for previous_step, next_step in zip(steps, steps[1:]):
            session.run(
                """
                MATCH (a:Step {id: $previous_id})
                MATCH (b:Step {id: $next_id})
                MERGE (a)-[:NEXT_STEP]->(b)
                """,
                previous_id=previous_step["id"],
                next_id=next_step["id"],
            )

        for fig in figures:
            session.run(
                """
                MERGE (f:Figure {path: $path})
                SET f.id           = $id,
                    f.label        = $label,
                    f.number       = $number,
                    f.caption_text = $caption_text,
                    f.doc          = $doc,
                    f.annotated    = false,
                    f.caption      = null,
                    f.ocr_text     = null,
                    f.entities     = [],
                    f.embedding    = null
                WITH f
                MATCH (d:Document {name: $doc})
                MERGE (d)-[:HAS_FIGURE]->(f)
                """,
                id=fig["id"],
                path=fig["path"],
                label=fig["label"],
                number=fig["number"],
                caption_text=fig["caption_text"],
                doc=fig["doc"],
            )

            if fig["step_id"]:
                session.run(
                    """
                    MATCH (f:Figure {path: $path})
                    MATCH (s:Step {id: $step_id})
                    MERGE (s)-[:HAS_FIGURE]->(f)
                    """,
                    path=fig["path"],
                    step_id=fig["step_id"],
                )

            link_concepts(session, "Figure", "path", fig["path"], f"{fig['label']} {fig['caption_text']}")

        for warning in warnings:
            session.run(
                """
                MERGE (w:Warning {id: $id})
                SET w.text = $text,
                    w.doc  = $doc
                WITH w
                MATCH (d:Document {name: $doc})
                MERGE (d)-[:HAS_WARNING]->(w)
                """,
                id=warning["id"],
                text=warning["text"],
                doc=warning["doc"],
            )

            if warning["step_id"]:
                session.run(
                    """
                    MATCH (w:Warning {id: $id})
                    MATCH (s:Step {id: $step_id})
                    MERGE (s)-[:HAS_WARNING]->(w)
                    """,
                    id=warning["id"],
                    step_id=warning["step_id"],
                )

            link_concepts(session, "Warning", "id", warning["id"], warning["text"])
            link_procedure_refs(session, "Warning", "id", warning["id"], warning["text"])

        figures_by_number = {fig["number"]: fig for fig in figures if fig.get("number")}
        for step in steps:
            step_text = f"{step['text']} {step.get('body') or ''}"
            figure_refs = set(re.findall(r"\(Figure\s+(\d+)\)", step_text, re.IGNORECASE))
            for figure_number in sorted(figure_refs):
                fig = figures_by_number.get(figure_number)
                if not fig:
                    continue
                session.run(
                    """
                    MATCH (s:Step {id: $step_id})
                    MATCH (f:Figure {path: $path})
                    MERGE (s)-[:MENTIONS_FIGURE]->(f)
                    """,
                    step_id=step["id"],
                    path=fig["path"],
                )

        with open(os.path.join(MARKDOWN_DIR, doc_name + ".md"), "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = text_splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_name}_chunk_{i}"
            session.run(
                """
                MERGE (c:TextChunk {id: $id})
                SET c.text = $text, c.doc = $doc
                WITH c MATCH (d:Document {name: $doc})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                id=chunk_id,
                text=chunk,
                doc=doc_name,
            )

            chunk_lower = chunk.lower()

            for step in steps:
                if step["text"].lower() in chunk_lower:
                    session.run(
                        """
                        MATCH (c:TextChunk {id: $chunk_id})
                        MATCH (s:Step {id: $step_id})
                        MERGE (c)-[:CONTAINS_STEP]->(s)
                        """,
                        chunk_id=chunk_id,
                        step_id=step["id"],
                    )

            for fig in figures:
                caption = fig["caption_text"].lower()
                if fig["label"].lower() in chunk_lower or (caption and caption in chunk_lower):
                    session.run(
                        """
                        MATCH (c:TextChunk {id: $chunk_id})
                        MATCH (f:Figure {path: $path})
                        MERGE (c)-[:CONTAINS_FIGURE]->(f)
                        """,
                        chunk_id=chunk_id,
                        path=fig["path"],
                    )

            for warning in warnings:
                warning_probe = warning["text"][:60].lower()
                if warning_probe and warning_probe in chunk_lower:
                    session.run(
                        """
                        MATCH (c:TextChunk {id: $chunk_id})
                        MATCH (w:Warning {id: $warning_id})
                        MERGE (c)-[:CONTAINS_WARNING]->(w)
                        """,
                        chunk_id=chunk_id,
                        warning_id=warning["id"],
                    )

            link_concepts(session, "TextChunk", "id", chunk_id, chunk)
            link_procedure_refs(session, "TextChunk", "id", chunk_id, chunk)

    print(f"  Steps:    {len(steps)}")
    print(f"  Figures:  {len(figures)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Chunks:   {len(chunks)}")


if __name__ == "__main__":
    if driver is None:
        raise RuntimeError("Install neo4j before building the graph: pip install -r config/requirements.txt")

    print("Building Knowledge Graph...\n")
    for filename in sorted(os.listdir(MARKDOWN_DIR)):
        if filename.endswith(".md"):
            filepath = os.path.join(MARKDOWN_DIR, filename)
            print(f"Processing: {filename}")
            doc_name, metadata, steps, figures, warnings = parse_markdown(filepath)
            build_graph(doc_name, metadata, steps, figures, warnings)
            print()

    with driver.session() as session:
        session.run("CREATE INDEX figure_path IF NOT EXISTS FOR (f:Figure) ON (f.path)")
        session.run("CREATE INDEX step_id IF NOT EXISTS FOR (s:Step) ON (s.id)")
        session.run("CREATE INDEX concept_id IF NOT EXISTS FOR (c:Concept) ON (c.id)")
        session.run("CREATE INDEX procedure_ref_code IF NOT EXISTS FOR (p:ProcedureRef) ON (p.code)")
        print("Indexes created.")

    driver.close()
    print("\nKnowledge Graph built successfully!")
