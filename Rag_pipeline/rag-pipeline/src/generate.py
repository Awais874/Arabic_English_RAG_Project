"""
Step 5 — Answer generation grounded in retrieved context (OpenAI API).

This module implements the generation stage of the Arabic-English
Retrieval-Augmented Generation (RAG) pipeline.

Pipeline:
    Question
        -> semantic or BM25 retrieval
        -> top-k document chunks
        -> OpenAI language model
        -> grounded answer with document citations

The language of the retrieved documents can be controlled separately
from the language of the question. This is required for the four
evaluation conditions:

    EN -> EN
    AR -> AR
    EN -> AR
    AR -> EN

The model is instructed to answer only from the retrieved context
rather than from its own external knowledge.

Requires an OpenAI API key stored in the OPENAI_API_KEY
environment variable.
"""

import os

from openai import OpenAI

import config
from retrieve import semantic_search, bm25_search


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """
You are an enterprise document question-answering assistant.

You must answer the user's question using ONLY the document excerpts
provided to you.

Follow these rules strictly:

1. Use only information contained in the provided document excerpts.

2. Do not use outside knowledge, assumptions, or information from your
   own training data.

3. If the provided excerpts do not contain enough information to
   answer the question, explicitly state that the available context
   does not contain enough information.

4. Do not invent facts, numbers, policies, procedures, names, dates,
   or other details.

5. Answer in the same language as the QUESTION:
      - English question -> English answer
      - Arabic question -> Arabic answer

   This rule applies even when the retrieved source documents are
   written in the other language.

6. At the end of the answer, provide a section called "Sources" for
   English answers or "المصادر" for Arabic answers.

7. In the source section, list only the document IDs that actually
   support the answer.

8. Document IDs must be copied exactly from the provided excerpts.

Your goal is to produce a concise, grounded answer that can be traced
back to the retrieved enterprise documents.
"""


# ============================================================
# Context construction
# ============================================================

def build_context_block(retrieved_chunks: list) -> str:
    """
    Convert retrieved chunks into a labelled context block for
    the language model.
    """

    lines = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        lines.append(
            f"[Excerpt {i}]\n"
            f"Document ID: {chunk['doc_id']}\n"
            f"Title: {chunk['title']}\n"
            f"Language: {chunk['language']}\n"
            f"Text:\n{chunk['text']}\n"
        )

    return "\n".join(lines)


# ============================================================
# OpenAI generation
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks: list,
    client=None
) -> str:
    """
    Generate an answer using only the retrieved document chunks.
    """

    if client is None:

        api_key = os.environ.get(
            config.OPENAI_API_KEY_ENV_VAR
        )

        if not api_key:
            raise RuntimeError(
                f"OpenAI API key not found. "
                f"Set the {config.OPENAI_API_KEY_ENV_VAR} "
                f"environment variable before running generation."
            )

        client = OpenAI(api_key=api_key)

    context_block = build_context_block(retrieved_chunks)

    user_message = f"""
DOCUMENT EXCERPTS

{context_block}

QUESTION

{question}

Answer the question according to the system instructions.
"""

    response = client.responses.create(
        model=config.OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_message,
    )

    return response.output_text


# ============================================================
# Complete RAG pipeline
# ============================================================

def answer_question(
    question: str,
    retrieval_method: str = "semantic",
    target_language: str = "en",
    top_k: int = config.TOP_K,
    client=None,
) -> dict:
    """
    Run the complete RAG pipeline for one question.

    Parameters
    ----------
    question:
        User question in English or Arabic.

    retrieval_method:
        "semantic" or "bm25".

    target_language:
        Language of the document collection to search:
        "en" or "ar".

    top_k:
        Number of chunks to retrieve.

    client:
        Optional OpenAI client. Useful when processing many
        questions because the same client can be reused.
    """

    if target_language not in ("en", "ar"):
        raise ValueError(
            "target_language must be 'en' or 'ar'"
        )

    if retrieval_method == "semantic":

        chunks = semantic_search(
            question,
            top_k=top_k,
            target_language=target_language,
        )

    elif retrieval_method == "bm25":

        chunks = bm25_search(
            question,
            top_k=top_k,
            target_language=target_language,
        )

    else:

        raise ValueError(
            "retrieval_method must be 'semantic' or 'bm25'"
        )

    answer = generate_answer(
        question,
        chunks,
        client=client,
    )

    return {
        "question": question,
        "retrieval_method": retrieval_method,
        "target_language": target_language,
        "retrieved_chunks": chunks,
        "answer": answer,
    }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    test_question = (
        "How many days of annual leave do employees get, "
        "and can unused days be carried over?"
    )

    result = answer_question(
        question=test_question,
        retrieval_method="semantic",
        target_language="en",
    )

    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(result["question"])

    print("\n" + "=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)

    for chunk in result["retrieved_chunks"]:
        print(
            f"{chunk['doc_id']} | "
            f"{chunk['language']} | "
            f"{chunk['title']}"
        )

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(result["answer"])