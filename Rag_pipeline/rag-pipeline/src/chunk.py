"""
Step 2 — Chunking.

Splits each extracted document into overlapping word-count chunks and
attaches metadata (doc_id, language, document_type, department, title)
looked up from the metadata CSV, so every chunk can be traced back to
its source document and section for citation and evaluation purposes.

This corresponds to the "chunking" and "metadata extraction" stages of
the conceptual framework in Chapter 3.

Run (after extract.py):
    python src/chunk.py
"""
import os
import re
import glob
import json
import pandas as pd

import config

DOC_ID_PATTERN = re.compile(r"^([A-Z]{2}-[A-Z]{3}-\d{2})")


def load_metadata() -> dict:
    """Returns {doc_id: {title, language, document_type, department, ...}}"""
    df = pd.read_csv(config.METADATA_CSV)
    df = df.set_index("doc_id")
    return df.to_dict(orient="index")


def extract_doc_id(filename: str) -> str:
    """Pulls e.g. 'EN-POL-01' out of 'EN-POL-01_IT_Acceptable_Use_Policy.txt'"""
    match = DOC_ID_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Could not parse doc_id from filename: {filename}")
    return match.group(1)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Simple sliding-window word-count chunker. Enterprise documents here
    are short (a few hundred to ~1000 words), so this keeps most
    documents to 2-6 chunks each, which is realistic for this document
    length and avoids over-fragmenting short policy clauses.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def main():
    metadata = load_metadata()
    txt_paths = glob.glob(os.path.join(config.EXTRACTED_DIR, "**", "*.txt"), recursive=True)

    if not txt_paths:
        print(f"No extracted .txt files found under {config.EXTRACTED_DIR}. Run extract.py first.")
        return

    all_chunks = []
    chunk_counter = 0

    for txt_path in txt_paths:
        filename = os.path.basename(txt_path)
        try:
            doc_id = extract_doc_id(filename)
        except ValueError as e:
            print(f"  Skipping {filename}: {e}")
            continue

        if doc_id not in metadata:
            print(f"  WARNING: {doc_id} not found in metadata CSV — skipping enrichment, using defaults")
            meta = {"title": doc_id, "language": "unknown", "document_type": "unknown", "department": "unknown"}
        else:
            meta = metadata[doc_id]

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        pieces = chunk_text(text, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)

        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc_id}_chunk{i:02d}",
                "doc_id": doc_id,
                "title": meta.get("title"),
                "language": meta.get("language"),
                "document_type": meta.get("document_type"),
                "department": meta.get("department"),
                "chunk_index": i,
                "text": piece,
            })
            chunk_counter += 1

    with open(config.CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {chunk_counter} chunks from {len(txt_paths)} documents.")
    print(f"Written to: {config.CHUNKS_JSON}")


if __name__ == "__main__":
    main()
