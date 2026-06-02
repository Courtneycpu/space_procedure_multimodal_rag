# Track 4: Vector search seeding localized multi-hop Cypher sweeps

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# 1. Connect to ChromaDB (Baseline)
CHROMA_DIR = os.path.abspath("data/chroma_baseline")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"}
)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# 2. Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
)

def retrieve_hybrid_context(query: str, top_k: int = 3):
    """Uses Vector Search to find the chunk, then Cypher to find connected Multimodal data."""
    
    # Step 1: Vector Search
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    
    if not docs:
        return []
    
    # Extract the chunk IDs (e.g., '1.101_chunk_4')
    chunk_ids = [d.metadata.get("id") for d in docs if "id" in d.metadata]
    
    enriched_context = []
    
    # Step 2: Graph Traversal
    with driver.session() as session:
        for cid in chunk_ids:
            res = session.run("""
                MATCH (c:TextChunk {id: $id})
                OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)-[:HAS_STEP]->(s:Step)
                OPTIONAL MATCH (s)-[:HAS_FIGURE]->(f:Figure)
                RETURN c.text AS chunk_text, 
                       c.doc AS doc, 
                       s.text AS step_text, 
                       s.number AS step_number,
                       f.path AS figure_path, 
                       f.caption AS llm_caption, 
                       f.ocr_text AS ocr_text
            """, id=cid)
            
            for r in res:
                enriched_context.append(dict(r))
                
    return enriched_context