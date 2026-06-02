# Track 2: Vector search over flattened markdown blocksimport os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 1. Connect to the ENRICHED Vector Store
CHROMA_DIR = os.path.abspath("data/chroma_enriched")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL, 
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR, 
    embedding_function=embeddings
)

def retrieve_enriched_context(query: str, top_k: int = 5):
    """Retrieves text chunks that include the flattened image annotations."""
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": top_k}
    )
    
    docs = retriever.invoke(query)
    
    return [{
        "doc": d.metadata.get("doc"),
        "chunk_id": d.metadata.get("id"),
        "text": d.page_content
    } for d in docs]