"""
Retrieval evaluation for the Arabic-English enterprise RAG dissertation.

Runs both retrieval methods over the same evaluation questions:
    - semantic: multilingual-e5-base + ChromaDB
    - lexical: BM25

The target_language field enforces the four controlled conditions:
    EN->EN, AR->AR, EN->AR, AR->EN

Metrics:
    Hit@1, Hit@3, Hit@5
    MRR
    Precision@5, Recall@5, F1@5

Run:
    python src/evaluate.py
"""

import os
import pandas as pd

from retrieve import semantic_search, bm25_search


QUESTIONS_PATH = os.path.join("data", "evaluation_questions_final.csv")
OUTPUT_DIR = "outputs"
TOP_K = 5


def parse_relevant_chunks(value):
    """Convert semicolon-separated ground-truth chunk IDs to a set."""
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


def evaluate_ranked_results(results, relevant_chunks):
    """Calculate ranked retrieval metrics for one query."""
    retrieved = [r["chunk_id"] for r in results]

    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved, start=1)
        if chunk_id in relevant_chunks
    ]

    hit_1 = int(any(rank <= 1 for rank in relevant_ranks))
    hit_3 = int(any(rank <= 3 for rank in relevant_ranks))
    hit_5 = int(any(rank <= 5 for rank in relevant_ranks))

    reciprocal_rank = (
        1.0 / min(relevant_ranks)
        if relevant_ranks
        else 0.0
    )

    retrieved_top5 = set(retrieved[:5])
    true_positives = len(retrieved_top5 & relevant_chunks)

    precision_5 = (
        true_positives / len(retrieved[:5])
        if retrieved[:5]
        else 0.0
    )

    recall_5 = (
        true_positives / len(relevant_chunks)
        if relevant_chunks
        else 0.0
    )

    f1_5 = (
        2 * precision_5 * recall_5 / (precision_5 + recall_5)
        if (precision_5 + recall_5) > 0
        else 0.0
    )

    first_relevant_rank = (
        min(relevant_ranks)
        if relevant_ranks
        else None
    )

    return {
        "hit_at_1": hit_1,
        "hit_at_3": hit_3,
        "hit_at_5": hit_5,
        "reciprocal_rank": reciprocal_rank,
        "precision_at_5": precision_5,
        "recall_at_5": recall_5,
        "f1_at_5": f1_5,
        "first_relevant_rank": first_relevant_rank,
    }


def run_method(method_name, search_function, questions):
    """Run one retrieval method over every evaluation row."""
    rows = []

    for number, (_, q) in enumerate(questions.iterrows(), start=1):
        relevant_chunks = parse_relevant_chunks(q["relevant_chunk_ids"])

        results = search_function(
            q["question"],
            top_k=TOP_K,
            target_language=q["target_language"]
        )

        metrics = evaluate_ranked_results(
            results,
            relevant_chunks
        )

        retrieved_ids = [
            r["chunk_id"]
            for r in results
        ]

        retrieved_docs = [
            r["doc_id"]
            for r in results
        ]

        rows.append({
            "question_id": q["question_id"],
            "condition": q["condition"],
            "query_language": q["query_language"],
            "target_language": q["target_language"],
            "document_type": q["document_type"],
            "method": method_name,
            "question": q["question"],
            "relevant_doc_id": q["relevant_doc_id"],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            "retrieved_chunk_ids": ";".join(retrieved_ids),
            "retrieved_doc_ids": ";".join(retrieved_docs),
            **metrics,
        })

        print(
            f"[{number:03d}/{len(questions)}] "
            f"{method_name:<8} "
            f"{q['question_id']} "
            f"{q['condition']} "
            f"Hit@1={metrics['hit_at_1']} "
            f"RR={metrics['reciprocal_rank']:.3f}"
        )

    return pd.DataFrame(rows)


def summarise(results, group_columns):
    """Calculate macro-average metrics for requested groups."""
    metric_columns = [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "reciprocal_rank",
        "precision_at_5",
        "recall_at_5",
        "f1_at_5",
    ]

    summary = (
        results
        .groupby(group_columns, as_index=False)[metric_columns]
        .mean()
    )

    summary["n_queries"] = (
        results
        .groupby(group_columns)
        .size()
        .values
    )

    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    questions = pd.read_csv(
        QUESTIONS_PATH,
        encoding="utf-8-sig"
    )

    required_columns = {
        "question_id",
        "condition",
        "query_language",
        "target_language",
        "question",
        "relevant_doc_id",
        "relevant_chunk_ids",
    }

    missing = required_columns - set(questions.columns)

    if missing:
        raise ValueError(
            f"Evaluation dataset is missing columns: {sorted(missing)}"
        )

    if len(questions) != 100:
        print(
            f"WARNING: expected 100 evaluation rows, "
            f"but found {len(questions)}."
        )

    print(f"Loaded {len(questions)} evaluation rows.")
    print()

    print("=" * 70)
    print("SEMANTIC RETRIEVAL")
    print("=" * 70)

    semantic_results = run_method(
        "semantic",
        semantic_search,
        questions
    )

    print()
    print("=" * 70)
    print("BM25 RETRIEVAL")
    print("=" * 70)

    bm25_results = run_method(
        "bm25",
        bm25_search,
        questions
    )

    detailed = pd.concat(
        [semantic_results, bm25_results],
        ignore_index=True
    )

    overall = summarise(
        detailed,
        ["method"]
    )

    by_condition = summarise(
        detailed,
        ["method", "condition"]
    )

    by_document_type = summarise(
        detailed,
        ["method", "document_type"]
    )

    detailed_path = os.path.join(
        OUTPUT_DIR,
        "retrieval_results_detailed.csv"
    )

    overall_path = os.path.join(
        OUTPUT_DIR,
        "retrieval_summary_overall.csv"
    )

    condition_path = os.path.join(
        OUTPUT_DIR,
        "retrieval_summary_by_condition.csv"
    )

    document_type_path = os.path.join(
        OUTPUT_DIR,
        "retrieval_summary_by_document_type.csv"
    )

    detailed.to_csv(
        detailed_path,
        index=False,
        encoding="utf-8-sig"
    )

    overall.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig"
    )

    by_condition.to_csv(
        condition_path,
        index=False,
        encoding="utf-8-sig"
    )

    by_document_type.to_csv(
        document_type_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(overall.to_string(index=False))

    print()
    print("=" * 70)
    print("RESULTS BY LANGUAGE CONDITION")
    print("=" * 70)
    print(by_condition.to_string(index=False))

    print()
    print("Files written:")
    print(f"  {detailed_path}")
    print(f"  {overall_path}")
    print(f"  {condition_path}")
    print(f"  {document_type_path}")


if __name__ == "__main__":
    main()
