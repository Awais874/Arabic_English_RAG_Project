# Arabic-English RAG Project

## About the Project


The project allows users to ask questions about Arabic and English PDF documents and receive answers based on the information in those documents.

The aim is to build and evaluate a system that can answer questions using information from Arabic and English PDF documents.
The project compares two document search methods, BM25 and semantic search, to understand how well they work with questions and documents in the same or different languages.

## Project Structure

- **Dissertation_DataSet/** – Arabic and English PDF documents used in the project.
- **Rag_pipeline/rag-pipeline/** – Python code, project data, evaluation files and results.

## Tools Used

- Python
- BM25
- Multilingual E5-base
- ChromaDB
- OCR
- OpenAI API

## How to Run the Project

### 1. Download the Project

Open a terminal and run:

```bash
git clone https://github.com/Awais874/Arabic_English_RAG_Project.git
cd Arabic_English_RAG_Project
cd Rag_pipeline/rag-pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the Requirements

```bash
pip install -r requirements.txt
```

### 4. Configure the API Key

Some parts of the project require an OpenAI API key.

Set the key in your environment before running answer generation. For example, in Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Do not upload your actual API key to GitHub.

### 5. Run the Project

The Python scripts are located in the `src` folder.

To process the documents:

```bash
python src/extract.py
python src/chunk.py
```

To create the search indexes:

```bash
python src/bm25_index.py
python src/embed_index.py
```

To run the evaluation:

```bash
python src/evaluate.py
```

The project also contains `retrieve.py` and `generate.py` for document retrieval and answer generation.

**Note:** These are the main scripts in the project. Depending on their configuration, some scripts may require additional arguments or existing input files.

## Dataset and Evaluation

The project uses 50 PDF documents:

- 25 English documents
- 25 Arabic documents

A total of 100 questions were evaluated across four language combinations:

- English to English
- Arabic to Arabic
- English to Arabic
- Arabic to English

## Results

| Evaluation | BM25 | Semantic Search |
|---|---|---|
| Hit@5 | 57% | 96% |
| MRR | 0.492 | 0.822 |

For answer generation, 96 out of 100 answers were fully correct according to the project's evaluation criteria.

