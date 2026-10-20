"""
Step 1 — Document extraction.

Extracts text from every PDF under data/raw_pdfs/ (recursively) and
writes one cleaned .txt file per source PDF into data/extracted_text/,
preserving the relative folder structure.

This is the first stage of the conceptual framework in Chapter 3:
    Documents -> Document Processing (extraction, cleaning) -> ...

IMPORTANT FINDING (worth reporting in your methodology/limitations
section, not just an implementation detail): standard text-layer
extraction (pdfplumber, and poppler's pdftotext) produces GARBLED,
letter-reordered output for the Arabic PDFs in this corpus, even
though the PDFs display correctly and are visually correct when
opened normally. This is a known interaction between WeasyPrint's
PDF text-layer encoding for shaped Arabic glyphs and extraction tools
that assume logical left-to-right character order when reconstructing
words. It is a genuine, citable instance of the "challenges applying
... retrieval techniques to Arabic-English document collections"
named in RQ3 — Arabic documents can defeat a text-extraction pipeline
that works perfectly for English source material from the same
system, for reasons that have nothing to do with the retrieval or
embedding stage.

The workaround used here: Arabic PDFs (identified by an "AR-" doc_id
prefix, or a language column in metadata) are extracted via OCR
(pytesseract, Arabic language pack) instead of the text layer. English
PDFs continue to use pdfplumber, which is faster and completely
reliable for them. This asymmetry is itself worth naming explicitly in
Chapter 4 rather than glossing over.

Requires (system packages, not pip — install via your OS package
manager): tesseract-ocr, tesseract-ocr-ara (Arabic language data),
poppler-utils (for pdf2image). See README.md for install commands.

Run:
    python src/extract.py
"""
import os
import re
import glob
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm

import config

OCR_DPI = 300  # higher DPI materially improves Arabic OCR accuracy vs the pdf2image default


def clean_text(raw_text: str) -> str:
    """
    Basic cleaning: collapse excessive whitespace/newlines introduced by
    extraction, without altering the actual words (important for
    Arabic, where over-aggressive cleaning can corrupt letter joining
    if done carelessly — here we only touch whitespace).
    """
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_arabic_doc(filename: str) -> bool:
    """Arabic documents in this corpus are named AR-XXX-NN_..."""
    return filename.upper().startswith("AR-")


def extract_pdf_text_layer(pdf_path: str) -> str:
    """Standard text-layer extraction — used for English PDFs."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
    return clean_text("\n\n".join(pages_text))


def extract_pdf_ocr(pdf_path: str, lang: str = "ara") -> str:
    """OCR-based extraction — used for Arabic PDFs (see module docstring for why)."""
    images = convert_from_path(pdf_path, dpi=OCR_DPI)
    pages_text = [pytesseract.image_to_string(img, lang=lang) for img in images]
    return clean_text("\n\n".join(pages_text))


def main():
    os.makedirs(config.EXTRACTED_DIR, exist_ok=True)
    pdf_paths = glob.glob(os.path.join(config.PDF_DIR, "**", "*.pdf"), recursive=True)

    if not pdf_paths:
        print(f"No PDFs found under {config.PDF_DIR}. "
              f"Copy your document PDFs there first (subfolders are fine).")
        return

    print(f"Found {len(pdf_paths)} PDFs. Extracting text "
          f"(Arabic via OCR, English via text layer)...")
    ocr_count, text_layer_count = 0, 0

    for pdf_path in tqdm(pdf_paths):
        filename = os.path.basename(pdf_path)
        rel_path = os.path.relpath(pdf_path, config.PDF_DIR)
        out_path = os.path.join(config.EXTRACTED_DIR, rel_path).replace(".pdf", ".txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            if is_arabic_doc(filename):
                text = extract_pdf_ocr(pdf_path)
                ocr_count += 1
            else:
                text = extract_pdf_text_layer(pdf_path)
                text_layer_count += 1
        except Exception as e:
            print(f"  FAILED on {rel_path}: {e}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Done. {text_layer_count} PDFs via text layer, {ocr_count} via OCR.")
    print(f"Extracted text written to: {config.EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
