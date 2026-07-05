"""Track 3: Entity-guided KG retrieval.

This version makes the first KG pipeline genuinely node-first:
1. Extract query entities with an LLM.
2. Link those entities directly to KG nodes across Document, Step, Figure,
   Warning, and TextChunk.
3. Use the best matched KG nodes as seeds.
4. Traverse typed KG relationships to collect connected context.

The public function `retrieve_kg_context(query, top_k=5)` is kept compatible
with the previous pipeline entry point.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT_DIR / "config" / ".env")

# Neo4j connection
# Keep the same environment-variable names as the rest of the project.
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    auth=(
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "12344321"),
    ),
)

# LLM client used only for query entity extraction.
client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL"),
    timeout=60,
)
MODEL = os.getenv("SAIA_DEFAULT_MODEL")

# Keep normal generation output clean. Set RAG_VERBOSE=1 only when debugging.
VERBOSE = os.getenv("RAG_VERBOSE", "0") == "1"


def debug_print(*args: Any, **kwargs: Any) -> None:
    if VERBOSE:
        print(*args, **kwargs)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "where", "which", "who", "why", "with", "about", "does",
    "do", "show", "shows", "explain", "describe", "procedure", "procedures",
    "step", "steps", "figure", "figures", "mention", "mentions", "mentioned",
}


def _parse_json_array(raw: str) -> list[str]:
    """Parse a JSON array from an LLM response, tolerating markdown fences."""
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)

    if isinstance(parsed, list):
        return [str(x) for x in parsed if str(x).strip()]

    # Tolerate a common alternative shape, e.g. {"entities": [...]}.
    if isinstance(parsed, dict):
        for key in ("entities", "terms", "keywords"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [str(x) for x in value if str(x).strip()]

    return []


def _fallback_terms_from_query(query: str) -> list[str]:
    """Simple fallback if the LLM entity extraction fails."""
    # Keep acronyms/numbers/technical tokens, remove common words.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-/]*|\d+(?:\.\d+)?", query)
    terms: list[str] = []
    for token in tokens:
        normalized = token.strip(" .,:;!?()[]{}\"'")
        if not normalized:
            continue
        if normalized.lower() in STOPWORDS:
            continue
        if len(normalized) < 2 and not normalized.isdigit():
            continue
        terms.append(normalized)
    return terms


def _normalize_entities(entities: list[str], query: str) -> list[str]:
    """Clean, deduplicate, and keep useful entity strings."""
    cleaned: list[str] = []

    for entity in entities:
        entity = re.sub(r"\s+", " ", str(entity)).strip(" .,:;!?()[]{}\"'")
        if not entity:
            continue
        if entity.lower() in STOPWORDS:
            continue
        if len(entity) < 2 and not entity.isdigit():
            continue
        cleaned.append(entity)

    # Only use deterministic fallback terms if the LLM returned nothing useful.
    # This avoids noisy generic words such as "perform", "one", or "needed" becoming KG seeds.
    if not cleaned:
        cleaned.extend(_fallback_terms_from_query(query))

    seen: set[str] = set()
    deduped: list[str] = []
    for entity in cleaned:
        key = entity.lower()
        if key not in seen:
            deduped.append(entity)
            seen.add(key)

    return deduped[:20]


def extract_entities(query: str) -> list[str]:
    """Extract key medical/procedural/equipment entities from the user query."""
    prompt = f"""Extract the key domain entities from this question.
Include medical terms, equipment names, procedure names, component names,
body parts, step numbers, figure numbers, acronyms, and important labels.
Return ONLY a JSON array of strings. No explanation, no markdown.
Example: ["EpiPen", "epinephrine", "injection", "Figure 2", "1.2"]

Question: {query}"""

    raw_entities: list[str] = []
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )
            raw = response.choices[0].message.content or ""
            raw_entities = _parse_json_array(raw)
            break
        except Exception as e:
            debug_print(f"  Entity extraction attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(3)

    return _normalize_entities(raw_entities, query)


def _find_seed_nodes(session: Any, entities: list[str], seed_k: int) -> list[dict[str, Any]]:
    """Link extracted query entities to concrete KG nodes across all node types."""
    result = session.run(
        """
        CALL () {
            MATCH (s:Step)
            WITH s, [e IN $entities WHERE
                toLower(coalesce(s.text, "")) CONTAINS toLower(e)
                OR toLower(coalesce(s.number, "")) = toLower(e)
            ] AS hits
            WHERE size(hits) > 0
            RETURN "Step" AS seed_type,
                   s.id AS seed_id,
                   s.doc AS doc,
                   s.text AS seed_text,
                   hits AS matched_entities,
                   size(hits) * 10 + 5 AS match_score

            UNION

            MATCH (f:Figure)
            WITH f, [e IN $entities WHERE
                toLower(coalesce(f.label, "")) CONTAINS toLower(e)
                OR toLower(coalesce(f.caption_text, "")) CONTAINS toLower(e)
                OR toLower(coalesce(f.caption, "")) CONTAINS toLower(e)
                OR toLower(coalesce(f.ocr_text, "")) CONTAINS toLower(e)
                OR toLower(coalesce(f.path, "")) CONTAINS toLower(replace(e, " ", "_"))
                OR any(x IN coalesce(f.entities, [])
                       WHERE trim(toString(x)) <> ""
                         AND (toLower(toString(x)) CONTAINS toLower(e)
                              OR toLower(e) CONTAINS toLower(toString(x))))
            ] AS hits
            WHERE size(hits) > 0
            RETURN "Figure" AS seed_type,
                   f.path AS seed_id,
                   f.doc AS doc,
                   coalesce(f.caption, f.caption_text, f.label, f.path) AS seed_text,
                   hits AS matched_entities,
                   size(hits) * 10 + 4 AS match_score

            UNION

            MATCH (d:Document)
            WITH d, [e IN $entities WHERE
                toLower(coalesce(d.name, "")) CONTAINS toLower(replace(e, " ", "_"))
                OR toLower(coalesce(d.title, "")) CONTAINS toLower(e)
                OR toLower(coalesce(d.objective, "")) CONTAINS toLower(e)
            ] AS hits
            WHERE size(hits) > 0
            RETURN "Document" AS seed_type,
                   d.name AS seed_id,
                   d.name AS doc,
                   coalesce(d.title, d.name) AS seed_text,
                   hits AS matched_entities,
                   size(hits) * 10 + 3 AS match_score

            UNION

            MATCH (w:Warning)
            WITH w, [e IN $entities WHERE
                toLower(coalesce(w.text, "")) CONTAINS toLower(e)
            ] AS hits
            WHERE size(hits) > 0
            RETURN "Warning" AS seed_type,
                   w.id AS seed_id,
                   w.doc AS doc,
                   w.text AS seed_text,
                   hits AS matched_entities,
                   size(hits) * 10 + 2 AS match_score

            UNION

            MATCH (c:TextChunk)
            WITH c, [e IN $entities WHERE
                toLower(coalesce(c.text, "")) CONTAINS toLower(e)
            ] AS hits
            WHERE size(hits) > 0
            RETURN "TextChunk" AS seed_type,
                   c.id AS seed_id,
                   c.doc AS doc,
                   left(c.text, 220) AS seed_text,
                   hits AS matched_entities,
                   size(hits) * 10 + 1 AS match_score
        }
        RETURN seed_type, seed_id, doc, seed_text, matched_entities, match_score
        ORDER BY match_score DESC, seed_type, doc, seed_id
        LIMIT $seed_k
        """,
        entities=entities,
        seed_k=seed_k,
    )
    return [dict(record) for record in result]


def _expand_step(session: Any, seed: dict[str, Any], per_seed_limit: int) -> list[dict[str, Any]]:
    result = session.run(
        """
        MATCH (s:Step {id: $seed_id})
        OPTIONAL MATCH (s)<-[:BELONGS_TO]-(c:TextChunk)
        OPTIONAL MATCH (s)-[:HAS_FIGURE]->(f:Figure)
        OPTIONAL MATCH (s)-[:HAS_WARNING]->(w:Warning)
        OPTIONAL MATCH (prev_s:Step)-[:NEXT_STEP]->(s)
        OPTIONAL MATCH (s)-[:NEXT_STEP]->(next_s:Step)
        RETURN c.text AS step_body,
               c.id AS chunk_id,
               c.chunk_index AS chunk_index,
               s.doc AS doc,
               s.text AS step_text,
               s.number AS step_number,
               f.path AS figure_path,
               coalesce(f.caption, f.caption_text) AS llm_caption,
               f.ocr_text AS ocr_text,
               f.entities AS figure_entities,
               w.text AS warning,
               prev_s.number AS previous_step_number,
               prev_s.text AS previous_step_text,
               next_s.number AS next_step_number,
               next_s.text AS next_step_text
        ORDER BY c.chunk_index, f.path
        LIMIT $limit
        """,
        seed_id=seed["seed_id"],
        limit=per_seed_limit,
    )
    return [dict(record) for record in result]


def _expand_figure(session: Any, seed: dict[str, Any], per_seed_limit: int) -> list[dict[str, Any]]:
    result = session.run(
        """
        MATCH (f:Figure {path: $seed_id})
        OPTIONAL MATCH (s:Step)-[:HAS_FIGURE]->(f)
        OPTIONAL MATCH (c1:TextChunk)-[:MENTIONS]->(f)
        OPTIONAL MATCH (c2:TextChunk)-[:BELONGS_TO]->(s)
        OPTIONAL MATCH (s)-[:HAS_WARNING]->(w:Warning)
        WITH f, s, w, coalesce(c1, c2) AS c
        RETURN c.text AS step_body,
               c.id AS chunk_id,
               c.chunk_index AS chunk_index,
               f.doc AS doc,
               s.text AS step_text,
               s.number AS step_number,
               f.path AS figure_path,
               coalesce(f.caption, f.caption_text) AS llm_caption,
               f.ocr_text AS ocr_text,
               f.entities AS figure_entities,
               w.text AS warning,
               null AS previous_step_number,
               null AS previous_step_text,
               null AS next_step_number,
               null AS next_step_text
        ORDER BY c.chunk_index
        LIMIT $limit
        """,
        seed_id=seed["seed_id"],
        limit=per_seed_limit,
    )
    return [dict(record) for record in result]


def _expand_document(session: Any, seed: dict[str, Any], per_seed_limit: int) -> list[dict[str, Any]]:
    """Expand a document seed, but only into chunks that match the query entities."""
    result = session.run(
        """
        MATCH (d:Document {name: $seed_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:TextChunk)
        WITH d, c
        WHERE c IS NOT NULL
          AND any(e IN $entities WHERE toLower(coalesce(c.text, "")) CONTAINS toLower(e))
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Step)
        OPTIONAL MATCH (c)-[:MENTIONS]->(mentioned_fig:Figure)
        OPTIONAL MATCH (s)-[:HAS_FIGURE]->(step_fig:Figure)
        OPTIONAL MATCH (s)-[:HAS_WARNING]->(w:Warning)
        OPTIONAL MATCH (prev_s:Step)-[:NEXT_STEP]->(s)
        OPTIONAL MATCH (s)-[:NEXT_STEP]->(next_s:Step)
        WITH d, c, s, coalesce(mentioned_fig, step_fig) AS f, w, prev_s, next_s
        RETURN c.text AS step_body,
               c.id AS chunk_id,
               c.chunk_index AS chunk_index,
               d.name AS doc,
               s.text AS step_text,
               s.number AS step_number,
               f.path AS figure_path,
               coalesce(f.caption, f.caption_text) AS llm_caption,
               f.ocr_text AS ocr_text,
               f.entities AS figure_entities,
               w.text AS warning,
               prev_s.number AS previous_step_number,
               prev_s.text AS previous_step_text,
               next_s.number AS next_step_number,
               next_s.text AS next_step_text
        ORDER BY c.chunk_index
        LIMIT $limit
        """,
        seed_id=seed["seed_id"],
        entities=seed.get("matched_entities", []),
        limit=per_seed_limit,
    )
    return [dict(record) for record in result]


def _expand_warning(session: Any, seed: dict[str, Any], per_seed_limit: int) -> list[dict[str, Any]]:
    result = session.run(
        """
        MATCH (w:Warning {id: $seed_id})
        OPTIONAL MATCH (s:Step)-[:HAS_WARNING]->(w)
        OPTIONAL MATCH (s)<-[:BELONGS_TO]-(c:TextChunk)
        OPTIONAL MATCH (s)-[:HAS_FIGURE]->(f:Figure)
        RETURN c.text AS step_body,
               c.id AS chunk_id,
               c.chunk_index AS chunk_index,
               w.doc AS doc,
               s.text AS step_text,
               s.number AS step_number,
               f.path AS figure_path,
               coalesce(f.caption, f.caption_text) AS llm_caption,
               f.ocr_text AS ocr_text,
               f.entities AS figure_entities,
               w.text AS warning,
               null AS previous_step_number,
               null AS previous_step_text,
               null AS next_step_number,
               null AS next_step_text
        ORDER BY c.chunk_index, f.path
        LIMIT $limit
        """,
        seed_id=seed["seed_id"],
        limit=per_seed_limit,
    )
    return [dict(record) for record in result]


def _expand_text_chunk(session: Any, seed: dict[str, Any], per_seed_limit: int) -> list[dict[str, Any]]:
    result = session.run(
        """
        MATCH (c:TextChunk {id: $seed_id})
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Step)
        OPTIONAL MATCH (c)-[:MENTIONS]->(mentioned_fig:Figure)
        OPTIONAL MATCH (s)-[:HAS_FIGURE]->(step_fig:Figure)
        OPTIONAL MATCH (s)-[:HAS_WARNING]->(w:Warning)
        OPTIONAL MATCH (prev_s:Step)-[:NEXT_STEP]->(s)
        OPTIONAL MATCH (s)-[:NEXT_STEP]->(next_s:Step)
        WITH c, s, coalesce(mentioned_fig, step_fig) AS f, w, prev_s, next_s
        RETURN c.text AS step_body,
               c.id AS chunk_id,
               c.chunk_index AS chunk_index,
               c.doc AS doc,
               s.text AS step_text,
               s.number AS step_number,
               f.path AS figure_path,
               coalesce(f.caption, f.caption_text) AS llm_caption,
               f.ocr_text AS ocr_text,
               f.entities AS figure_entities,
               w.text AS warning,
               prev_s.number AS previous_step_number,
               prev_s.text AS previous_step_text,
               next_s.number AS next_step_number,
               next_s.text AS next_step_text
        ORDER BY c.chunk_index, f.path
        LIMIT $limit
        """,
        seed_id=seed["seed_id"],
        limit=per_seed_limit,
    )
    return [dict(record) for record in result]


EXPANDERS = {
    "Step": _expand_step,
    "Figure": _expand_figure,
    "Document": _expand_document,
    "Warning": _expand_warning,
    "TextChunk": _expand_text_chunk,
}


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Stable key used to remove duplicates created by multi-edge traversal."""
    return (
        row.get("doc"),
        row.get("chunk_id"),
        row.get("step_number"),
        row.get("figure_path"),
        row.get("warning"),
    )


def retrieve_kg_context(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve KG context using entity-guided node linking and graph traversal.

    Args:
        query: Natural-language user question.
        top_k: Number of seed KG nodes to use. Each seed can return multiple
               connected context rows.

    Returns:
        A list of context dictionaries. The original fields from the old
        implementation are preserved where possible:
        step_body, chunk_index, doc, step_text, step_number, figure_path,
        llm_caption, ocr_text, figure_entities, warning.

        Extra debug fields are added:
        seed_type, seed_id, seed_text, matched_entities, match_score.
    """
    entities = extract_entities(query)
    if not entities:
        print(f"  Extracted entities: []", flush=True)
        return []

    print(f"  Extracted entities: {entities}", flush=True)

    # Use top_k seed nodes; each seed can contribute a few rows.
    # Keeping per_seed_limit modest prevents noisy graph expansion.
    seed_k = max(top_k, 1)
    per_seed_limit = 4

    with driver.session() as session:
        seeds = _find_seed_nodes(session, entities, seed_k=seed_k)
        if not seeds:
            debug_print("  No KG seed nodes matched the extracted entities.")
            return []

        debug_print("  KG seed nodes:")
        for seed in seeds:
            debug_print(
                f"    - {seed['seed_type']} | {seed['seed_id']} "
                f"| score={seed['match_score']} | hits={seed['matched_entities']}"
            )

        rows: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        for seed in seeds:
            expander = EXPANDERS.get(seed["seed_type"])
            if expander is None:
                continue

            expanded_rows = expander(session, seed, per_seed_limit=per_seed_limit)
            for row in expanded_rows:
                row = dict(row)
                row.update(
                    {
                        "seed_type": seed["seed_type"],
                        "seed_id": seed["seed_id"],
                        "seed_text": seed["seed_text"],
                        "matched_entities": seed["matched_entities"],
                        "match_score": seed["match_score"],
                    }
                )

                key = _row_key(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    return rows


if __name__ == "__main__":
    test_query = "What figures or steps mention epinephrine injection?"
    context = retrieve_kg_context(test_query, top_k=5)
    print(f"\nRetrieved {len(context)} context rows")
    for i, row in enumerate(context[:5], start=1):
        print(f"\n--- Row {i} ---")
        print(json.dumps(row, indent=2, default=str))
