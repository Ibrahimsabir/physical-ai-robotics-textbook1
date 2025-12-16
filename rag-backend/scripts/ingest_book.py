import os
from app.rag.loader import load_markdown
from app.rag.chunker import chunk_text
from app.rag.embeddings import embed_and_store

BOOK_DOCS_PATH = "../speckit-ebook/docs"

for root, _, files in os.walk(BOOK_DOCS_PATH):
    for file in files:
        if file.endswith(".md") or file.endswith(".mdx"):
            path = os.path.join(root, file)
            text = load_markdown(path)
            chunks = chunk_text(text)
            embed_and_store(chunks, source=file)

print("✅ Book content indexed successfully")
