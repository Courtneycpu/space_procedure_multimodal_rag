# Track 3: Explicit entity matching via deterministic Cypher

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parents[2] / "config" / ".env")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12344321"))
)

def retrieve_kg_context(query_entities: list):
    if not query_entities:
        return []

    with driver.session() as session:
        result = session.run("""
            MATCH (s:Step)
            WHERE any(entity IN $entities
                WHERE toLower(s.text) CONTAINS toLower(entity)
                   OR toLower(s.doc)  CONTAINS toLower(entity))

            // Also pull sibling steps from the same document for richer context
            WITH collect(DISTINCT s.doc) AS matched_docs
            MATCH (s2:Step)
            WHERE s2.doc IN matched_docs
            OPTIONAL MATCH (s2)-[:HAS_WARNING]->(w:Warning)

            RETURN s2.text     AS step_text,
                   s2.number   AS step_number,
                   s2.doc      AS doc,
                   w.text      AS warning
            ORDER BY s2.doc, s2.number
        """, entities=query_entities)

        return [dict(r) for r in result]