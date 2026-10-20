"""
Retrieval functions for the two experimental branches:

1. semantic_search()
   Multilingual-E5 dense semantic retrieval using ChromaDB.

2. bm25_search()
   BM25 lexical retrieval using separate English and Arabic indexes.

Both retrieval methods can be restricted to the target document
language for controlled multilingual evaluation.
"""

import pickle

import chromadb
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------
# Cached resources
# ---------------------------------------------------------

_embedding_model = None
_chroma_collection = None
_bm25_data = None


def _get_embedding_model():
    """Load multilingual E5 only once."""
    global _embedding_model

    if _embedding_model is None:
        print(
            f"Loading embedding model: "
            f"{config.EMBEDDING_MODEL_NAME}"
        )
        _embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME
        )

    return _embedding_model


def _get_chroma_collection():
    """Connect to persistent ChromaDB collection."""
    global _chroma_collection

    if _chroma_collection is None:
        client = chromadb.PersistentClient(
            path=config.CHROMA_DIR
        )

        _chroma_collection = client.get_collection(
            config.CHROMA_COLLECTION_NAME
        )

    return _chroma_collection


def _get_bm25_data():
    """Load separate English and Arabic BM25 indexes."""
    global _bm25_data

    if _bm25_data is None:
        with open(
            config.BM25_INDEX_PATH,
            "rb"
        ) as f:
            _bm25_data = pickle.load(f)

    return _bm25_data


# ---------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------

def semantic_search(
    query: str,
    top_k: int = config.TOP_K,
    target_language: str = None
) -> list:
    """
    Retrieve chunks using multilingual-E5.

    target_language should normally be:
        "en" or "ar"
    """

    model = _get_embedding_model()
    collection = _get_chroma_collection()

    query_embedding = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    query_arguments = {
        "query_embeddings": query_embedding,
        "n_results": top_k
    }

    if target_language:
        query_arguments["where"] = {
            "language": target_language
        }

    results = collection.query(
        **query_arguments
    )

    output = []

    for i in range(
        len(results["ids"][0])
    ):
        meta = results["metadatas"][0][i]

        # Chroma uses cosine distance.
        similarity_score = (
            1.0 -
            results["distances"][0][i]
        )

        output.append({
            "chunk_id":
                results["ids"][0][i],

            "doc_id":
                meta["doc_id"],

            "title":
                meta["title"],

            "language":
                meta["language"],

            "document_type":
                meta["document_type"],

            "text":
                results["documents"][0][i],

            "score":
                float(similarity_score),
        })

    return output


# ---------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------

def bm25_search(
    query: str,
    top_k: int = config.TOP_K,
    target_language: str = None
) -> list:
    """
    Retrieve chunks using BM25.

    English and Arabic use completely separate BM25 indexes.
    This means BM25 corpus statistics are calculated only from
    the target-language corpus.

    target_language:
        "en" or "ar"
    """

    from bm25_index import simple_tokenize

    data = _get_bm25_data()

    # Evaluation always provides a target language.
    # This check prevents accidentally mixing the corpora.
    if target_language not in ("en", "ar"):
        raise ValueError(
            "BM25 retrieval requires "
            "target_language='en' or 'ar'."
        )

    # Select the correct independent language index.
    language_data = data[target_language]

    bm25 = language_data["bm25"]
    chunks = language_data["chunks"]

    tokenized_query = simple_tokenize(
        query
    )

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    output = []

    for idx in ranked_indices:

        chunk = chunks[idx]

        output.append({
            "chunk_id":
                chunk["chunk_id"],

            "doc_id":
                chunk["doc_id"],

            "title":
                chunk["title"],

            "language":
                chunk["language"],

            "document_type":
                chunk["document_type"],

            "text":
                chunk["text"],

            "score":
                float(scores[idx]),
        })

    return output


# ---------------------------------------------------------
# Manual sanity check
# ---------------------------------------------------------

if __name__ == "__main__":

    test_queries = [
        {
            "question":
                "How many days of annual leave do employees get?",
            "target_language":
                "en"
        },
        {
            "question":
                "كم عدد أيام الإجازة السنوية التي يستحقها الموظف؟",
            "target_language":
                "ar"
        }
    ]

    for test in test_queries:

        question = test["question"]
        target_language = test[
            "target_language"
        ]

        print()
        print("=" * 70)
        print(
            f"Query: {question}"
        )
        print(
            f"Target corpus: "
            f"{target_language}"
        )
        print("=" * 70)

        print(
            "\n--- Semantic retrieval: Top 3 ---"
        )

        semantic_results = semantic_search(
            question,
            top_k=3,
            target_language=target_language
        )

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):
            print(
                f"{rank}. "
                f"[{result['score']:.3f}] "
                f"{result['doc_id']} "
                f"({result['language']})"
            )

            print(
                f"   "
                f"{result['text'][:120]}..."
            )

        print(
            "\n--- BM25 retrieval: Top 3 ---"
        )

        bm25_results = bm25_search(
            question,
            top_k=3,
            target_language=target_language
        )

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):
            print(
                f"{rank}. "
                f"[{result['score']:.3f}] "
                f"{result['doc_id']} "
                f"({result['language']})"
            )

            print(
                f"   "
                f"{result['text'][:120]}..."
            )