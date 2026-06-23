"""
src/ingestion/build_vector_store.py
===================================
Builds two separate ChromaDB vector stores directly from the markdown files:
1. Baseline Store: From data/raw_markdown/
2. Enriched Store: From data/enriched_markdown/ (Flattened images)

Done , create 107 chunks , enriched embading with document name and type, and save to chroma_baseline directory.
No noise chunks found, all chunks are meaningful and above the 100 character threshold.

"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / "config" / ".env")

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

# get rid of any noise chunks that are too short to be meaningful (e.g. "Figure 1: ...") by merging them 
# with the previous chunk until we have a minimum length of 100 characters. 
# This is important because very short chunks can lead to poor retrieval performance and noisy results.
def merge_short_chunks(chunks: list[str], min_length: int = 100) -> list[str]:
    merged = []
    for chunk in chunks:
        if merged and len(chunk.strip()) < min_length:
            merged[-1] = merged[-1] + " " + chunk.strip()
        else:
            merged.append(chunk)
    return merged

# For parsing the document name and extracting rich metadata, we can use the following helper functions.
def get_doc_type(procedure_number: str, raw_name: str) -> str:
    category = procedure_number.split(".")[0]
    
    if category == "1":
        return "Emergency Procedure"
    elif category == "2":
        # Reference/anatomy docs get a different label
        if any(x in raw_name for x in ["ANATOMY", "EXAM"]):
            return "Dental Reference"
        elif "TREATMENT" in raw_name:
            return "Dental Treatment"
        else:
            return "Dental Procedure"
    return "Medical Procedure"

def parse_doc_name(doc_name: str) -> dict:
    parts = doc_name.split("_", 1)
    procedure_number = parts[0]
    raw_name = parts[1] if len(parts) > 1 else doc_name

    procedure_name = raw_name.replace("_-_", ": ").replace("_", " ").title()
    category_id = procedure_number.split(".")[0]
    doc_type = get_doc_type(procedure_number, raw_name)

    # Strip redundant doc_type prefix from procedure_name
    # e.g. "Dental Procedure: Nerve Block" → "Nerve Block"
    if procedure_name.lower().startswith(doc_type.lower() + ": "):
        procedure_name = procedure_name[len(doc_type) + 2:]

    return {
        "procedure_number": procedure_number,
        "procedure_name": procedure_name,
        "category_id": category_id,
        "doc_type": doc_type,
    }

# build the vector store directly from the markdown files This way we can enrich the text that gets embedded
#  with metadata like the document name and type, which can help improve retrieval relevance. 
# use merge_short_chunks to ensure we don't have any noisy chunks that are too short to be meaningful.
def build_store_from_directory(source_dir: Path, persist_dir: str, collection_name: str):
    """Reads markdown files, chunks them, and saves to a persistent Chroma directory."""
    if not source_dir.exists() or not list(source_dir.glob("*.md")):
        print(f"Skipping {collection_name}: No markdown files found in {source_dir}")
        return

    print(f"\nReading markdown files from {source_dir}...")
    
    langchain_docs = []
    
    for md_file in sorted(source_dir.glob("*.md")):
        doc_name = md_file.stem
        content = md_file.read_text(encoding='utf-8')
        
        # Parse rich metadata from filename
        parsed = parse_doc_name(doc_name)
        
        chunks = text_splitter.split_text(content)
        chunks = merge_short_chunks(chunks, min_length=100)
        
        for i, chunk_text in enumerate(chunks):
            
            # Enrich the text that gets embedded
            text_to_embed = (
                f"{parsed['doc_type']}: {parsed['procedure_name']}\n\n"
                f"{chunk_text}"
            )
            
            doc = Document(
                page_content=text_to_embed,          # add the document name to the text being embedded
                metadata={
                    "doc": doc_name,
                    "id": f"{doc_name}_chunk_{i}", 
                    "procedure_number": parsed["procedure_number"],
                    "procedure_name": parsed["procedure_name"],
                    "doc_type": parsed["doc_type"],
                    "category_id": parsed["category_id"],
                    "chunk_index": i,
                }
            )
            langchain_docs.append(doc)

    print(f"Embedding {len(langchain_docs)} chunks...")

    ids = [doc.metadata["id"] for doc in langchain_docs]

    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        ids=ids,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    print(f"✅ Success! {collection_name} is ready at: {persist_dir}")

# We can call build_store_from_directory twice to create both the baseline and enriched vector stores.
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


# just call the class on what you want to build a vector store, and we can easily switch 
# between building the baseline vs enriched store by commenting/uncommenting the relevant lines in main().
if __name__ == "__main__":
    main()