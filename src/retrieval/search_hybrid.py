# Track 4: Vector search seeding localized KG traversal for multimodal context

import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[2] / "config" / ".env")

ROOT_DIR   = Path(__file__).parents[2]
CHROMA_DIR = str(ROOT_DIR / "data" / "chroma_baseline")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12344321"))
)


def retrieve_hybrid_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Step 1 — vector search finds the most relevant text chunks.
    Step 2 — KG traversal expands each chunk to linked Steps and Figures.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    docs = retriever.invoke(query)

    if not docs:
        return []

    chunk_ids = [d.metadata.get("id") for d in docs if "id" in d.metadata]

    enriched = []
    with driver.session() as session:
        for cid in chunk_ids:
            result = session.run("""
                MATCH (c:TextChunk {id: $id})
                OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)-[:HAS_STEP]->(s:Step)
                OPTIONAL MATCH (s)-[:HAS_FIGURE]->(f:Figure)
                OPTIONAL MATCH (s)-[:HAS_WARNING]->(w:Warning)
                RETURN c.text      AS chunk_text,
                       c.doc       AS doc,
                       s.text      AS step_text,
                       s.number    AS step_number,
                       f.path      AS figure_path,
                       f.caption   AS llm_caption,
                       f.ocr_text  AS ocr_text,
                       w.text      AS warning
            """, id=cid)
            enriched.extend(dict(r) for r in result)

    return enriched