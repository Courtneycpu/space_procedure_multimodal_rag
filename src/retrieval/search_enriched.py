# Track 2: Vector search over flattened markdown blocksimport os
from anyio import Path
from langchain_chroma import Chroma
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Connect to the ENRICHED Vector Store
CHROMA_DIR = str(Path(__file__).parents[2] / "data" / "chroma_enriched")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL, 
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="enriched_text_chunks"   # add this line
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

if __name__ == "__main__":
    count = vectorstore._collection.count()
    print(f"Documents in chroma_enriched: {count}")
    
    if count == 0:
        # Check if baseline exists instead
        baseline = str(Path(__file__).parents[2] / "data" / "chroma_baseline")
        vs_base = Chroma(persist_directory=baseline, embedding_function=embeddings)
        print(f"Documents in chroma_baseline: {vs_base._collection.count()}")