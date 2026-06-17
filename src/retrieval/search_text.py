# Track 1: Vector search over plain text chunks

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Connect to the Baseline Vector Store
CHROMA_DIR = os.path.abspath("data/chroma_baseline")
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

def retrieve_text_context(query: str, top_k: int = 5):
    """Retrieves the most relevant pure text chunks."""
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": top_k}
    )
    
    docs = retriever.invoke(query)
    
    # Format the LangChain documents into clean dictionaries
    return [{
        "doc": d.metadata.get("doc"),
        "chunk_id": d.metadata.get("id"),
        "text": d.page_content
    } for d in docs]