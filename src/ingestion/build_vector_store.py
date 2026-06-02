"""
src/ingestion/build_vector_store.py
===================================
Builds two separate ChromaDB vector stores directly from the markdown files:
1. Baseline Store: From data/raw_markdown/
2. Enriched Store: From data/enriched_markdown/ (Flattened images)
"""

import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
RAW_MD_DIR = Path("data/raw_markdown")
ENRICHED_MD_DIR = Path("data/enriched_markdown")

CHROMA_BASELINE_DIR = "data/chroma_baseline"
CHROMA_ENRICHED_DIR = "data/chroma_enriched"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# ──────────────────────────────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL, 
    model_kwargs={"device": "cpu"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

def build_store_from_directory(source_dir: Path, persist_dir: str, collection_name: str):
    """Reads markdown files, chunks them, and saves to a persistent Chroma directory."""
    if not source_dir.exists() or not list(source_dir.glob("*.md")):
        print(f"⚠️ Skipping {collection_name}: No markdown files found in {source_dir}")
        return

    print(f"\n📥 Reading markdown files from {source_dir}...")
    
    langchain_docs = []
    
    # We read file-by-file so we can explicitly align the Chunk IDs with Neo4j
    for md_file in sorted(source_dir.glob("*.md")):
        doc_name = md_file.stem
        content = md_file.read_text(encoding='utf-8')
        
        # Split text into chunks
        chunks = text_splitter.split_text(content)
        
        for i, chunk_text in enumerate(chunks):
            # Create a LangChain Document with explicit matching IDs
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "doc": doc_name,
                    "id": f"{doc_name}_chunk_{i}"  # Crucial for Hybrid Search!
                }
            )
            langchain_docs.append(doc)

    print(f"⚙️ Embedding {len(langchain_docs)} chunks...")

    # Extract our exact IDs so Chroma uses them instead of random UUIDs
    ids = [doc.metadata["id"] for doc in langchain_docs]

    # Build and persist the vector database
    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        ids=ids,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    print(f"✅ Success! {collection_name} is ready at: {persist_dir}")

def main():
    print("=== Building Vector Databases ===")
    
    # 1. Build Track 1 (Pure Text Baseline)
    build_store_from_directory(
        source_dir=RAW_MD_DIR, 
        persist_dir=CHROMA_BASELINE_DIR,
        collection_name="raw_text_chunks"
    )

    # 2. Build Track 2 (Enriched / Flattened Baseline)
    build_store_from_directory(
        source_dir=ENRICHED_MD_DIR, 
        persist_dir=CHROMA_ENRICHED_DIR,
        collection_name="enriched_text_chunks"
    )

if __name__ == "__main__":
    main()