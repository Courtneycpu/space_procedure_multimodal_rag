# Track 3: Explicit entity matching via deterministic Cypher

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
)

def retrieve_kg_context(query_entities: list):
    """
    Searches the graph using exact entities (e.g., ['EpiPen', 'CPR']).
    Retrieves connected steps, figures, and warnings.
    """
    if not query_entities:
        return []

    with driver.session() as session:
        # Your teammate's excellent Cypher traversal query
        result = session.run("""
            MATCH (s:Step)
            WHERE any(entity IN $entities WHERE toLower(s.text) CONTAINS toLower(entity))
            OPTIONAL MATCH (s)-[:HAS_FIGURE]->(f:Figure)
            OPTIONAL MATCH (s)-[:HAS_WARNING]->(w:Warning)
            RETURN s.text AS step_text,
                   s.number AS step_number,
                   s.doc AS doc,
                   f.path AS figure_path,
                   f.caption AS llm_caption,
                   f.ocr_text AS ocr_text,
                   w.text AS warning
        """, entities=query_entities)
        
        return [dict(r) for r in result]