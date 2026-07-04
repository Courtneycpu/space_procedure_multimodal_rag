# Track 2: Vector search over flattened markdown blocks
import os 
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

NEIGHBOR_CHUNKS_BEFORE = 1
NEIGHBOR_CHUNKS_AFTER = 3


def _neighbor_ids(doc_name: str, chunk_index: int) -> list[str]:
    start = max(0, chunk_index - NEIGHBOR_CHUNKS_BEFORE)
    stop = chunk_index + NEIGHBOR_CHUNKS_AFTER
    return [f"{doc_name}_chunk_{i}" for i in range(start, stop + 1)]


def _format_doc(text: str | None, metadata: dict | None) -> dict:
    metadata = metadata or {}
    return {
        "doc": metadata.get("doc"),
        "chunk_id": metadata.get("id"),
        "chunk_index": metadata.get("chunk_index"),
        "text": text or "",
    }


def retrieve_enriched_context(query: str, top_k: int = 5):
    """Retrieves enriched chunks plus nearby chunks with figure descriptions."""
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": top_k}
    )
    
    docs = retriever.invoke(query)

    expanded_ids = []
    seen_ids = set()
    for doc in docs:
        metadata = doc.metadata
        doc_name = metadata.get("doc")
        chunk_index = metadata.get("chunk_index")

        if doc_name is None or chunk_index is None:
            chunk_id = metadata.get("id")
            if chunk_id and chunk_id not in seen_ids:
                expanded_ids.append(chunk_id)
                seen_ids.add(chunk_id)
            continue

        for chunk_id in _neighbor_ids(doc_name, int(chunk_index)):
            if chunk_id not in seen_ids:
                expanded_ids.append(chunk_id)
                seen_ids.add(chunk_id)

    if not expanded_ids:
        return [_format_doc(d.page_content, d.metadata) for d in docs]

    collection = vectorstore._collection
    expanded = collection.get(ids=expanded_ids)
    by_id = {
        chunk_id: (text, metadata)
        for chunk_id, text, metadata in zip(
            expanded["ids"],
            expanded["documents"],
            expanded["metadatas"],
        )
    }

    return [
        _format_doc(*by_id[chunk_id])
        for chunk_id in expanded_ids
        if chunk_id in by_id
    ]

if __name__ == "__main__":
    count = vectorstore._collection.count()
    print(f"Documents in chroma_enriched: {count}")
    
    if count == 0:
        # Check if baseline exists instead
        baseline = str(Path(__file__).parents[2] / "data" / "chroma_baseline")
        vs_base = Chroma(persist_directory=baseline, embedding_function=embeddings)
        print(f"Documents in chroma_baseline: {vs_base._collection.count()}")
