"""
Step 3 — Embedding generation and vector database indexing (semantic branch).

Embeds every chunk using a multilingual sentence embedding model
(intfloat/multilingual-e5-base) and stores the vectors, text, and
metadata in a local ChromaDB collection.

This is the "Multilingual Embedding Generation -> Vector Database"
branch of the conceptual framework in Chapter 3.

The semantic retrieval branch is kept separate from the BM25 keyword
retrieval branch so that the two retrieval approaches can be compared
independently during evaluation.

The multilingual-e5 model uses prefix instructions:
    passage: <document chunk>
    query: <user query>

Document embeddings are L2-normalized and ChromaDB uses cosine distance
for semantic similarity.

Run:
    python src/embed_index.py
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


def main():
    # ---------------------------------------------------------
    # 1. Load the document chunks
    # ---------------------------------------------------------
    with open(config.CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found. Run chunk.py first.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    # ---------------------------------------------------------
    # 2. Load multilingual E5 embedding model
    # ---------------------------------------------------------
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    print(
        "(The first run downloads the model. "
        "This may take some time and only needs to happen once.)"
    )

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    # ---------------------------------------------------------
    # 3. Create persistent ChromaDB client
    # ---------------------------------------------------------
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    # Delete the old collection if it already exists.
    # This prevents stale vectors from previous experiments.
    try:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
        print("Existing ChromaDB collection removed.")
    except Exception:
        pass

    # Explicitly use cosine distance for semantic similarity.
    collection = client.create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------
    print(f"Embedding {len(chunks)} chunks...")

    batch_size = 32

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]

        # Multilingual-E5 convention:
        # document passages must begin with "passage: "
        texts_with_prefix = [
            f"passage: {chunk['text']}"
            for chunk in batch
        ]

        # Normalize embeddings so cosine comparison is consistent.
        embeddings = model.encode(
            texts_with_prefix,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        # -----------------------------------------------------
        # 5. Store vectors, text and metadata in ChromaDB
        # -----------------------------------------------------
        collection.add(
            ids=[
                chunk["chunk_id"]
                for chunk in batch
            ],

            embeddings=embeddings,

            documents=[
                chunk["text"]
                for chunk in batch
            ],

            metadatas=[
                {
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"] or "",
                    "language": chunk["language"] or "",
                    "document_type": chunk["document_type"] or "",
                    "department": chunk["department"] or "",
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in batch
            ],
        )

    # ---------------------------------------------------------
    # 6. Confirm successful indexing
    # ---------------------------------------------------------
    print()
    print("Embedding and indexing complete.")
    print(f"Chunks indexed: {collection.count()}")
    print(f"ChromaDB location: {config.CHROMA_DIR}")
    print(f"Collection: {config.CHROMA_COLLECTION_NAME}")
    print("Similarity metric: cosine")
    print("Embedding normalization: enabled")


if __name__ == "__main__":
    main()