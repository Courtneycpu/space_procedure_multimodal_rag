import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password123")
)

print("Adding new relationships to KG...\n")

with driver.session() as session:

    # 1. NEXT_STEP: link steps in sequential order within a document
    print("Adding NEXT_STEP relationships...")
    session.run("""
        MATCH (d:Document)-[:HAS_STEP]->(s:Step)
        WITH d, s ORDER BY s.number
        WITH d, collect(s) AS steps
        UNWIND range(0, size(steps)-2) AS i
        WITH steps[i] AS s1, steps[i+1] AS s2
        MERGE (s1)-[:NEXT_STEP]->(s2)
    """)
    print("  ✓ Done\n")

    # 2. MENTIONS: link text chunks to figures they reference
    print("Adding MENTIONS relationships...")
    result = session.run("""
        MATCH (c:TextChunk)
        RETURN c.id AS id, c.text AS text
    """)
    chunks = [dict(r) for r in result]

    for chunk in chunks:
        # Find figure references in chunk text
        fig_matches = re.findall(
            r'images/[^\)"\s]+\.png', chunk['text']
        )
        for fig_path in fig_matches:
            session.run("""
                MATCH (c:TextChunk {id: $chunk_id})
                MATCH (f:Figure {path: $path})
                MERGE (c)-[:MENTIONS]->(f)
            """, chunk_id=chunk['id'], path=fig_path)

    print("  ✓ Done\n")

    # 3. BELONGS_TO: link text chunks to steps they contain
    print("Adding BELONGS_TO relationships...")
    result = session.run("""
        MATCH (c:TextChunk)
        RETURN c.id AS id, c.text AS text, c.doc AS doc
    """)
    chunks = [dict(r) for r in result]

    result2 = session.run("""
        MATCH (s:Step)
        RETURN s.id AS id, s.text AS text, s.doc AS doc
    """)
    steps = [dict(r) for r in result2]

    for chunk in chunks:
        for step in steps:
            if (chunk['doc'] == step['doc'] and
                    step['text'].lower()[:30] in
                    chunk['text'].lower()):
                session.run("""
                    MATCH (c:TextChunk {id: $chunk_id})
                    MATCH (s:Step {id: $step_id})
                    MERGE (c)-[:BELONGS_TO]->(s)
                """,
                chunk_id=chunk['id'],
                step_id=step['id'])

    print("  ✓ Done\n")

    # 4. RELATED_TO: link documents that reference each other
    print("Adding RELATED_TO relationships...")
    result = session.run("""
        MATCH (d:Document)
        RETURN d.name AS name
    """)
    docs = [r['name'] for r in result]

    MARKDOWN_DIR = os.path.expanduser("~/project11/data/markdown/")
    for doc_name in docs:
        filepath = os.path.join(MARKDOWN_DIR, doc_name + '.md')
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Find references to other procedures
        for other_doc in docs:
            if other_doc != doc_name:
                # Check if other doc number appears in text
                doc_number = other_doc.split('_')[0]
                if doc_number in text:
                    session.run("""
                        MATCH (d1:Document {name: $doc1})
                        MATCH (d2:Document {name: $doc2})
                        MERGE (d1)-[:RELATED_TO]->(d2)
                    """, doc1=doc_name, doc2=other_doc)

    print("  ✓ Done\n")

    # 5. WARNS_ABOUT: link warnings to nearby figures
    print("Adding WARNS_ABOUT relationships...")
    session.run("""
        MATCH (s:Step)-[:HAS_WARNING]->(w:Warning)
        MATCH (s)-[:HAS_FIGURE]->(f:Figure)
        MERGE (w)-[:WARNS_ABOUT]->(f)
    """)
    print("  ✓ Done\n")

    # Print summary of all relationships
    print("=" * 50)
    print("RELATIONSHIP SUMMARY:")
    result = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS relationship, count(r) AS count
        ORDER BY count DESC
    """)
    for r in result:
        print(f"  {r['relationship']}: {r['count']}")

driver.close()
print("\nAll relationships added successfully!")