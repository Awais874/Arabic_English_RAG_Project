# Arabic-English RAG Pipeline — Setup and Usage

This is a working implementation of the conceptual framework from Chapter 3:
document processing -> parallel semantic (dense) and lexical (BM25)
retrieval branches -> LLM generation with citations -> evaluation.

## 1. Prerequisites

- Python 3.10 or later
- An Anthropic API key (https://console.anthropic.com) for the generation step
  (retrieval-only steps do NOT need this key)
- System packages for Arabic OCR (see "Important finding" below for why this
  is needed): `tesseract-ocr`, `tesseract-ocr-ara` (Arabic language data),
  and `poppler-utils`. Only needed if you re-run `extract.py` yourself —
  pre-extracted text is already included, see Section 3.

  ```bash
  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr tesseract-ocr-ara poppler-utils
  # macOS (Homebrew)
  brew install tesseract tesseract-lang poppler
  ```

## 2. Setup

```bash
cd rag-pipeline
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Anthropic API key as an environment variable (never hardcode it in the scripts):

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"      # Mac/Linux, current terminal session
setx ANTHROPIC_API_KEY "sk-ant-your-key-here"         # Windows (open a new terminal after)
```

## 3. Your documents (already included)

The 50-document corpus, its metadata CSV, pre-extracted text, and the
pre-built BM25 index are ALREADY included under `data/` — verified working
end-to-end before handoff (see "What's been tested" below). You can go
straight to Section 5 to try retrieval immediately.

To add more documents later (e.g. extending toward 60-70), drop new PDFs
into `data/raw_pdfs/` following the same `AR-`/`EN-` doc_id filename
prefix convention, add their rows to `data/document_metadata.csv`, then
re-run the pipeline from Step 4 below.

## 4. Run the pipeline, in order

```bash
python src/extract.py        # PDFs -> cleaned .txt files (data/extracted_text/)
python src/chunk.py          # .txt files -> chunks.json, enriched with metadata
python src/embed_index.py    # chunks -> embeddings -> ChromaDB (semantic branch)
python src/bm25_index.py     # chunks -> BM25 index (lexical baseline branch)
```

The first run of `embed_index.py` downloads the embedding model (~1.1GB) —
this only happens once and is cached afterward.

### Important finding from building this: Arabic PDF text extraction

Standard text-layer extraction (pdfplumber, and poppler's `pdftotext`)
produces GARBLED, letter-reordered output for the Arabic PDFs in this
corpus — even though the PDFs display perfectly when opened normally.
This turned out to be an artefact of how the PDF-generation tool
(WeasyPrint) encodes shaped Arabic glyphs in the PDF's text layer:
visually correct, but not reliably machine-extractable by tools that
assume logical left-to-right character order when reconstructing words.

`extract.py` works around this by routing Arabic PDFs (identified by
their `AR-` doc_id prefix) through OCR (`pytesseract`, Arabic language
pack) instead of the text layer, while English PDFs keep using the
faster, fully reliable text-layer extraction. This is worth reporting
as a genuine finding in your Chapter 4 methodology or limitations
section — it's a concrete, first-hand instance of the "challenges
applying ... techniques to Arabic-English document collections" your
RQ3 asks about, and it wasn't something either of us anticipated before
actually running the extraction. If you later source real-world Arabic
PDFs (scanned or exported from Word) rather than these synthetic ones,
re-test whether the same issue occurs — it may be specific to this
PDF-generation tool rather than a general Arabic-PDF phenomenon, and
that distinction matters for how you frame the finding.

## 5. Try it out

Quick sanity check comparing both retrieval branches on a couple of test queries:

```bash
python src/retrieve.py
```

Full question -> retrieval -> generated answer, using the semantic branch:

```bash
python src/generate.py
```

## 6. Once your evaluation question set exists

Build `data/evaluation_questions.csv` with columns:
`question_id, question, expected_doc_ids, language`

(`expected_doc_ids` is semicolon-separated if a question needs more than
one source document, e.g. `EN-POL-02;EN-MAN-02`)

Then run:

```bash
python src/evaluate.py --eval_csv data/evaluation_questions.csv
```

This prints average precision/recall/F1 for the semantic branch vs the
BM25 branch — your core evidence for RQ1 — and writes full per-question
results (including generated answers and retrieved context) to
`outputs/evaluation_results.json`, which is also the input format RAGAS
expects if you install it separately for the faithfulness/answer-relevancy
metrics discussed in Chapter 3.

## 7. What to change if you want to compare embedding models

`src/config.py` has a single `EMBEDDING_MODEL_NAME` constant. To compare
two models (e.g. for a richer RQ3 discussion), rerun `embed_index.py`
after changing this constant, pointing `CHROMA_DIR` and
`CHROMA_COLLECTION_NAME` to a second location so both indexes coexist,
and add a second retrieval function in `retrieve.py` mirroring
`semantic_search()`.

## What's been tested before handoff

Everything except the two steps that need external network access this
sandbox didn't have were run end-to-end against your real 50-document
corpus and verified:
- PDF extraction (both the OCR branch and text-layer branch) — verified
  correct on real Arabic and English output
- Chunking — 153 chunks produced from the 50 documents
- BM25 indexing and retrieval — tested with real English and Arabic
  queries against the real corpus; correctly retrieved the right source
  document every time
- ChromaDB add/query mechanics — verified structurally with stand-in
  vectors of the correct shape (the actual `multilingual-e5-base` model
  couldn't be downloaded in the sandbox this was built in, so run
  `embed_index.py` yourself as the first real test of the semantic branch)
- Prompt/context formatting for generation — verified the exact text
  Claude will receive is well-formed

Not yet tested (needs your machine/API key): the actual embedding model
download and encoding, and the live Claude API call. Both are
straightforward — try `python src/generate.py` first, since a failure
there is most likely a missing/incorrect `ANTHROPIC_API_KEY`.

## Known limitations to mention in your dissertation

- The BM25 tokeniser (`bm25_index.py`) is a naive whitespace/regex
  tokeniser — no Arabic-specific normalisation (hamza/alef variants,
  definite article stripping) or stemming. This likely understates
  BM25's true achievable performance on Arabic text relative to a
  production system using an Arabic-aware tokeniser (e.g. CAMeL Tools).
  Worth flagging explicitly in your limitations section rather than
  letting it pass silently, since it affects how you interpret an
  "RAG beats BM25 on Arabic" result.
- Chunking is a fixed word-count sliding window, not sentence- or
  section-aware. For short policy documents like these this rarely
  splits mid-sentence badly, but it's a simplification worth naming.
