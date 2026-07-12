"""the class used mainly to retrive chunks from the DB to check for quality"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

PERSIST_DIR = "data/chroma_baseline" # data/chroma_enriched or data/chroma_baseline

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="row_text_chunks" # enriched_text_chunks or row_text_chunks
)

# Get the raw collection
collection = vectorstore._collection

# Returns everything — ids, documents, metadata
# all_chunks = collection.get()

# all_chunks["ids"]       → list of chunk IDs e.g. "1.102_AED_ASSISTED_CPR_chunk_4"
# all_chunks["documents"] → list of chunk texts
# all_chunks["metadatas"] → list of metadata dicts

"""""
chunks = collection.get(where={"doc": "1.102_AED_ASSISTED_CPR"})
for text in chunks["documents"]:
    print(text[:200])
    print("---")  

chunks = collection.get(where={"doc": "1.102_AED_ASSISTED_CPR"})
for text in chunks["documents"]:
    print(text[:200])
    print("---")

chunk = collection.get(ids=["1.102_AED_ASSISTED_CPR_chunk_4"])
print(chunk["documents"][0])

print(collection.count())
"""
import json

all_chunks = collection.get()
if all_chunks is None:
    all_chunks = {"ids": [], "metadatas": [], "documents": []}

ids = all_chunks.get("ids") or []
metadatas = all_chunks.get("metadatas") or []
documents = all_chunks.get("documents") or []

data = []

for i in range(len(ids)):
    data.append({
        "id": ids[i],
        "metadata": metadatas[i] if i < len(metadatas) else {},
        "text": documents[i] if i < len(documents) else "",
    })

with open("chunks_dump_enriched.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Saved to chunks_dump.json")


