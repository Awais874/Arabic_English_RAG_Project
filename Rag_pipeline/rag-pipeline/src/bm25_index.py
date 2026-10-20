"""
BM25 keyword baseline.

Builds two independent BM25 indexes:
    - English corpus
    - Arabic corpus

This ensures BM25 statistics are calculated only from the target-language
corpus used in each evaluation condition.
"""

import json
import pickle
import re

from rank_bm25 import BM25Okapi

import config


ARABIC_DIACRITICS = re.compile(
    r"""
    ّ |
    َ |
    ً |
    ُ |
    ٌ |
    ِ |
    ٍ |
    ْ |
    ـ
    """,
    re.VERBOSE,
)


def normalize_arabic(text: str) -> str:
    """Apply lightweight Arabic orthographic normalisation."""

    text = re.sub(ARABIC_DIACRITICS, "", text)

    # Normalise Alef variants
    text = re.sub(r"[إأآٱ]", "ا", text)

    # Alef Maqsura -> Ya
    text = text.replace("ى", "ي")

    return text


def simple_tokenize(text: str) -> list:
    """Tokenise English or Arabic text for BM25."""

    text = text.lower()
    text = normalize_arabic(text)

    return re.findall(
        r"\w+",
        text,
        flags=re.UNICODE
    )


def build_language_index(chunks, language):
    """
    Build a BM25 index using only chunks from one language.
    """

    language_chunks = [
        chunk
        for chunk in chunks
        if chunk["language"] == language
    ]

    tokenized_corpus = [
        simple_tokenize(chunk["text"])
        for chunk in language_chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "bm25": bm25,
        "chunks": language_chunks
    }


def main():

    with open(
        config.CHUNKS_JSON,
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found. Run chunk.py first.")
        return

    print(f"Total chunks loaded: {len(chunks)}")

    # Build completely separate BM25 indexes.
    english_index = build_language_index(
        chunks,
        "en"
    )

    arabic_index = build_language_index(
        chunks,
        "ar"
    )

    payload = {
        "en": english_index,
        "ar": arabic_index
    }

    with open(
        config.BM25_INDEX_PATH,
        "wb"
    ) as f:
        pickle.dump(payload, f)

    print()
    print("BM25 indexing complete.")
    print(
        f"English chunks indexed: "
        f"{len(english_index['chunks'])}"
    )
    print(
        f"Arabic chunks indexed: "
        f"{len(arabic_index['chunks'])}"
    )
    print(
        f"Total chunks: "
        f"{len(english_index['chunks']) + len(arabic_index['chunks'])}"
    )
    print(
        f"BM25 index location: "
        f"{config.BM25_INDEX_PATH}"
    )
    print("Separate English and Arabic BM25 indexes: enabled")
    print("Arabic normalization: enabled")
    print("Arabic stemming: disabled")


if __name__ == "__main__":
    main()