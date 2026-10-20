# Dissertation Dataset — Full Corpus (50 documents)

Synthetic Arabic-English enterprise document corpus for the dissertation
"Designing and Evaluating a Multilingual Arabic-English Retrieval-Augmented
Generation System for Enterprise Document Question Answering."

## Structure

```
raw/
  english/   (25 documents)
    policy/     EN-POL-01 to EN-POL-07  (7 documents)
    manual/     EN-MAN-01 to EN-MAN-07  (7 documents)
    report/     EN-REP-01 to EN-REP-06  (6 documents)
    contract/   EN-CON-01 to EN-CON-05  (5 documents)
  arabic/    (25 documents)
    policy/     AR-POL-01 to AR-POL-07  (7 documents)
    manual/     AR-MAN-01 to AR-MAN-07  (7 documents)
    report/     AR-REP-01 to AR-REP-06  (6 documents)
    contract/   AR-CON-01 to AR-CON-05  (5 documents)
metadata/
  document_metadata.csv   -- catalogue with doc_id, title, language, type,
                             department, source, dates, processing status
```

PDF versions of the same 50 documents are provided separately, generated
via weasyprint with correct RTL layout, Arabic script shaping (Noto Naskh
Arabic), and table column ordering for the Arabic files.

## Status

- 50 of the target 50-70 documents complete (25 English, 25 Arabic; all
  four document types represented in both languages, matching the
  proposal's specified document categories: policies, manuals, reports,
  and sample contracts)
- All documents are clearly synthetic (fictional company "Meridian
  Holdings Group") — no real organisational or personal data
- Documents cross-reference each other by Document ID (e.g. the
  onboarding handbook references the leave policy's ID), which is useful
  for testing multi-hop retrieval and citation accuracy
- Metadata catalogue matches the fields specified in the proposal's
  Research Data Management Plan
- Evaluation question set intentionally NOT yet created — to be built
  once the document processing pipeline exists, per the agreed sequencing

## Next steps

- Optionally extend further toward the 60-70 upper end of the proposal's
  target range if more variety is wanted
- Build the document processing pipeline (extraction, cleaning, chunking,
  embedding) — the PDFs are the right input format for this, since a real
  enterprise collection would rarely be clean markdown
- After the pipeline exists: construct the 50-100 question evaluation set
  against the actual indexed content
