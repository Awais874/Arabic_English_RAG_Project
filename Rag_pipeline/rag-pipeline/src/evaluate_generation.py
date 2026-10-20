"""
Generation evaluation for the Arabic-English enterprise RAG dissertation.

This script evaluates the answer-generation stage of the semantic RAG
pipeline using the existing controlled evaluation dataset.

For every evaluation row it:

    1. Reads the question and target document language.
    2. Performs semantic retrieval using multilingual-e5-base.
    3. Retrieves the top-5 chunks.
    4. Sends the retrieved context to the OpenAI generation model.
    5. Saves the generated answer alongside the expected answer.
    6. Records whether the expected document/chunk was retrieved.
    7. Records whether the generated answer cites the expected document.

The script supports a small test run before the full experiment.

Test:
    python src/evaluate_generation.py --limit 4

Full experiment:
    python src/evaluate_generation.py
"""

import os
import sys
import argparse
import pandas as pd

from openai import OpenAI

import config
from retrieve import semantic_search
from generate import generate_answer


QUESTIONS_PATH = os.path.join(
    config.PROJECT_ROOT,
    "data",
    "evaluation_questions_final.csv"
)

OUTPUT_PATH = os.path.join(
    config.OUTPUTS_DIR,
    "generation_results_detailed.csv"
)

TOP_K = config.TOP_K


def parse_relevant_chunks(value):
    """
    Convert semicolon-separated ground-truth chunk IDs to a set.
    """

    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


def create_openai_client():
    """
    Create one OpenAI client that can be reused for all questions.
    """

    api_key = os.environ.get(
        config.OPENAI_API_KEY_ENV_VAR
    )

    if not api_key:
        raise RuntimeError(
            f"OpenAI API key not found.\n"
            f"Set the {config.OPENAI_API_KEY_ENV_VAR} "
            f"environment variable before running this script."
        )

    return OpenAI(api_key=api_key)


def evaluate_one_question(row, client):
    """
    Run semantic retrieval and answer generation for one
    evaluation question.
    """

    question = row["question"]
    target_language = row["target_language"]

    relevant_doc_id = str(
        row["relevant_doc_id"]
    ).strip()

    relevant_chunks = parse_relevant_chunks(
        row["relevant_chunk_ids"]
    )

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    retrieved = semantic_search(
        question,
        top_k=TOP_K,
        target_language=target_language,
    )

    retrieved_chunk_ids = [
        chunk["chunk_id"]
        for chunk in retrieved
    ]

    retrieved_doc_ids = [
        chunk["doc_id"]
        for chunk in retrieved
    ]

    # --------------------------------------------------------
    # Retrieval indicators
    # --------------------------------------------------------

    relevant_chunk_retrieved = int(
        any(
            chunk_id in relevant_chunks
            for chunk_id in retrieved_chunk_ids
        )
    )

    relevant_doc_retrieved = int(
        relevant_doc_id in retrieved_doc_ids
    )

    first_relevant_chunk_rank = None

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1
    ):
        if chunk_id in relevant_chunks:
            first_relevant_chunk_rank = rank
            break

    # --------------------------------------------------------
    # Answer generation
    # --------------------------------------------------------

    generated_answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved,
        client=client,
    )

    # --------------------------------------------------------
    # Citation indicator
    #
    # This is intentionally simple and transparent:
    # does the generated response contain the expected
    # supporting document ID?
    # --------------------------------------------------------

    expected_doc_cited = int(
        relevant_doc_id in generated_answer
    )

    return {
        "question_id": row["question_id"],
        "condition": row["condition"],
        "query_language": row["query_language"],
        "target_language": target_language,
        "document_type": row.get(
            "document_type",
            ""
        ),
        "question": question,
        "expected_answer": row["expected_answer"],
        "relevant_doc_id": relevant_doc_id,
        "relevant_chunk_ids": row["relevant_chunk_ids"],
        "retrieved_chunk_ids": ";".join(
            retrieved_chunk_ids
        ),
        "retrieved_doc_ids": ";".join(
            retrieved_doc_ids
        ),
        "relevant_chunk_retrieved": (
            relevant_chunk_retrieved
        ),
        "relevant_doc_retrieved": (
            relevant_doc_retrieved
        ),
        "first_relevant_chunk_rank": (
            first_relevant_chunk_rank
        ),
        "generated_answer": generated_answer,
        "expected_doc_cited": expected_doc_cited,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer generation for the "
            "Arabic-English RAG system."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N evaluation rows. "
            "Useful for testing before the full experiment."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------------

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
        "expected_answer",
        "relevant_doc_id",
        "relevant_chunk_ids",
    }

    missing = required_columns - set(
        questions.columns
    )

    if missing:
        raise ValueError(
            "Evaluation dataset is missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Optional test limit
    # --------------------------------------------------------

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        questions = questions.head(
            args.limit
        ).copy()

    total = len(questions)

    print("=" * 70)
    print("RAG GENERATION EVALUATION")
    print("=" * 70)

    print(
        f"Questions to process: {total}"
    )

    print(
        f"Retrieval method: semantic"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    print(
        f"Generation model: {config.OPENAI_MODEL}"
    )

    if args.limit is not None:
        print(
            f"TEST MODE: first {total} questions only"
        )
    else:
        print(
            "FULL EXPERIMENT MODE"
        )

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # OpenAI client
    # --------------------------------------------------------

    client = create_openai_client()

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    rows = []

    for number, (_, row) in enumerate(
        questions.iterrows(),
        start=1
    ):

        print(
            f"[{number:03d}/{total}] "
            f"{row['question_id']} "
            f"{row['condition']} ..."
        )

        try:

            result = evaluate_one_question(
                row,
                client,
            )

            rows.append(result)

            print(
                f"    Relevant chunk retrieved: "
                f"{result['relevant_chunk_retrieved']}"
            )

            print(
                f"    Relevant document retrieved: "
                f"{result['relevant_doc_retrieved']}"
            )

            print(
                f"    Expected document cited: "
                f"{result['expected_doc_cited']}"
            )

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

            # Save an error row rather than losing
            # the rest of the experiment.

            rows.append({
                "question_id": row["question_id"],
                "condition": row["condition"],
                "query_language": row["query_language"],
                "target_language": row["target_language"],
                "document_type": row.get(
                    "document_type",
                    ""
                ),
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "relevant_doc_id": row["relevant_doc_id"],
                "relevant_chunk_ids": row[
                    "relevant_chunk_ids"
                ],
                "retrieved_chunk_ids": "",
                "retrieved_doc_ids": "",
                "relevant_chunk_retrieved": "",
                "relevant_doc_retrieved": "",
                "first_relevant_chunk_rank": "",
                "generated_answer": "",
                "expected_doc_cited": "",
                "error": str(exc),
            })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        config.OUTPUTS_DIR,
        exist_ok=True
    )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Display basic summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATION RUN COMPLETE")
    print("=" * 70)

    print(
        f"Rows processed: {len(results)}"
    )

    successful = results[
        results["generated_answer"]
        .fillna("")
        .astype(str)
        .str.len() > 0
    ]

    print(
        f"Answers generated successfully: "
        f"{len(successful)}"
    )

    if len(successful) > 0:

        chunk_rate = pd.to_numeric(
            successful["relevant_chunk_retrieved"],
            errors="coerce"
        ).mean()

        doc_rate = pd.to_numeric(
            successful["relevant_doc_retrieved"],
            errors="coerce"
        ).mean()

        citation_rate = pd.to_numeric(
            successful["expected_doc_cited"],
            errors="coerce"
        ).mean()

        print(
            f"Relevant chunk retrieval rate: "
            f"{chunk_rate:.3f}"
        )

        print(
            f"Relevant document retrieval rate: "
            f"{doc_rate:.3f}"
        )

        print(
            f"Expected-document citation rate: "
            f"{citation_rate:.3f}"
        )

    print()
    print(
        f"Results saved to:\n{OUTPUT_PATH}"
    )

    print()
    print(
        "NOTE: This file records generated answers and "
        "ground-truth answers for subsequent answer-quality "
        "evaluation. Citation presence alone is not treated "
        "as proof that an answer is correct."
    )


if __name__ == "__main__":
    main()