"""
Shared configuration for the Arabic-English RAG pipeline.
Edit paths/constants here rather than scattering them across scripts.
"""
import os


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw_pdfs"
)

METADATA_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "document_metadata.csv"
)

EXTRACTED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "extracted_text"
)

CHUNKS_JSON = os.path.join(
    PROJECT_ROOT,
    "data",
    "chunks.json"
)

CHROMA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "chroma_db"
)

BM25_INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "bm25_index.pkl"
)

OUTPUTS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


# ============================================================
# Chunking
# ============================================================

# Target number of words per chunk.
CHUNK_SIZE_WORDS = 180

# Overlap between neighbouring chunks to reduce information
# loss at chunk boundaries.
CHUNK_OVERLAP_WORDS = 30


# ============================================================
# Embedding
# ============================================================

# Multilingual embedding model used for Arabic-English
# semantic retrieval.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# E5 models use:
#   "passage: " prefix when indexing document chunks
#   "query: " prefix when embedding user questions


# ============================================================
# Retrieval
# ============================================================

# Number of chunks returned for each query.
# The same value is used for semantic retrieval and BM25
# to ensure a fair comparison.
TOP_K = 5


# ============================================================
# Generation (OpenAI)
# ============================================================

# OpenAI model used to generate answers from retrieved context.
OPENAI_MODEL = "gpt-5.6-luna"

# Name of the Windows environment variable containing
# the OpenAI API key.
# Never hardcode the actual API key in this file.
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"


# ============================================================
# ChromaDB
# ============================================================

# Collection containing the multilingual document embeddings.
CHROMA_COLLECTION_NAME = "enterprise_docs_ar_en"