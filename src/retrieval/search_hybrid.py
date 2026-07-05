# Track 4: Vector search seeding localized KG traversal for multimodal context

import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[2] / "config" / ".env")

CHROMA_DIR = str(Path(__file__).parents[2] / "data" / "chroma_baseline")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="raw_text_chunks"
)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12344321"))
)


def retrieve_hybrid_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Track 4:
    1. Use Chroma vector search to find relevant TextChunk IDs.
    2. Use each TextChunk as a seed node in Neo4j.
    3. Traverse only the local KG neighborhood:
       TextChunk -> BELONGS_TO -> Step
       TextChunk -> MENTIONS -> Figure
       Step -> HAS_FIGURE -> Figure
       Step -> HAS_WARNING -> Warning
       Step -> NEXT_STEP -> next Step
       previous Step -> NEXT_STEP -> Step
    """

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    docs = retriever.invoke(query)

    if not docs:
        return []

    # Get Chroma chunk IDs.
    # This assumes your vector-store metadata contains "id".
    chunk_ids = []
    for d in docs:
        cid = d.metadata.get("id")
        if cid:
            chunk_ids.append(cid)

    if not chunk_ids:
        print("No chunk IDs found in Chroma metadata.")
        print("Example metadata:", docs[0].metadata)
        return []

    enriched = []

    with driver.session() as session:
        for cid in chunk_ids:
            result = session.run(
                """
                MATCH (c:TextChunk {id: $id})

                OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)

                // Prefer the real KG link between chunk and step
                OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Step)

                // Fallback only if BELONGS_TO is missing
                OPTIONAL MATCH (d)-[:HAS_STEP]->(fallback_s:Step)
                WHERE s IS NULL
                  AND toLower(c.text) CONTAINS toLower(left(fallback_s.text, 30))

                WITH c, d, coalesce(s, fallback_s) AS s

                OPTIONAL MATCH (prev:Step)-[:NEXT_STEP]->(s)
                OPTIONAL MATCH (s)-[:NEXT_STEP]->(next:Step)

                OPTIONAL MATCH (c)-[:MENTIONS]->(mentioned_fig:Figure)
                OPTIONAL MATCH (s)-[:HAS_FIGURE]->(step_fig:Figure)

                WITH c, d, s, prev, next,
                     coalesce(mentioned_fig, step_fig) AS f

                OPTIONAL MATCH (s)-[:HAS_WARNING]->(w:Warning)

                RETURN
                    c.id AS chunk_id,
                    c.text AS chunk_text,
                    c.doc AS doc,
                    c.chunk_index AS chunk_index,

                    d.title AS document_title,
                    d.objective AS document_objective,

                    s.id AS step_id,
                    s.text AS step_text,
                    s.number AS step_number,

                    prev.text AS previous_step_text,
                    prev.number AS previous_step_number,

                    next.text AS next_step_text,
                    next.number AS next_step_number,

                    f.path AS figure_path,
                    f.caption_text AS original_caption,
                    f.caption AS llm_caption,
                    f.ocr_text AS ocr_text,
                    f.entities AS figure_entities,

                    w.text AS warning
                ORDER BY c.doc, c.chunk_index, s.number
                """,
                id=cid
            )

            enriched.extend(dict(r) for r in result)

    return enriched