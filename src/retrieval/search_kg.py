# Track 3: Explicit entity matching via deterministic Cypher using enteties extraction

import os
import json
import time
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parents[2] / "config" / ".env")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12344321"))
)

client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL"),
    timeout=60,
)
MODEL = os.getenv("SAIA_DEFAULT_MODEL")


def extract_entities(query: str) -> list[str]:
    """Uses LLM to extract key medical terms and equipment names from the query."""
    prompt = f"""Extract the key medical terms, equipment names, procedure names,
and component names from this question.
Return ONLY a JSON array of strings. No explanation, no markdown.
Example: ["EpiPen", "epinephrine", "injection"]

Question: {query}"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"  Entity extraction attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(3)
    return []


def retrieve_kg_context(query: str, top_k = 5) -> list[dict]:
    """
    Extracts entities from the query, then retrieves matching steps,
    figures, and warnings from the KG.
    """
    entities = extract_entities(query)
    if not entities:
        return []

    print(f"  Extracted entities: {entities}")

    with driver.session() as session:
        result = session.run("""
            MATCH (c:TextChunk)
            WHERE any(entity IN $entities
                WHERE toLower(c.text) CONTAINS toLower(entity)
                   OR toLower(coalesce(c.procedure_name, "")) CONTAINS toLower(entity)
                   OR toLower(replace(c.doc, "_", " ")) CONTAINS toLower(entity))
            WITH c
            ORDER BY c.doc, c.chunk_index
            LIMIT $top_k
            OPTIONAL MATCH (d:Document {name: c.doc})-[:HAS_STEP]->(candidate:Step)
            WITH c,
                 [step IN collect(candidate)
                  WHERE step IS NOT NULL
                    AND (toLower(c.text) CONTAINS toLower(step.text)
                         OR toLower(c.text) CONTAINS toLower(left(step.text, 30)))]
                 AS matched_steps
            UNWIND CASE
                WHEN size(matched_steps) = 0 THEN [null]
                ELSE matched_steps
            END AS s
            OPTIONAL MATCH (s)-[:HAS_FIGURE]->(f:Figure)
            OPTIONAL MATCH (s)-[:HAS_WARNING]->(w:Warning)
            RETURN c.text         AS step_body,
                   c.chunk_index  AS chunk_index,
                   c.doc          AS doc,
                   s.text         AS step_text,
                   s.number       AS step_number,
                   f.path         AS figure_path,
                   f.caption      AS llm_caption,
                   f.ocr_text     AS ocr_text,
                   w.text         AS warning
            ORDER BY c.doc, c.chunk_index, s.number
        """, entities=entities, top_k=top_k)

        return [dict(r) for r in result]
