# PDF RAG + RAGAS Evaluation App

This project creates a complete Retrieval-Augmented Generation pipeline:

1. Upload a PDF.
2. Parse PDFs into structured documents and Markdown using Docling.
3. Create token-aware semantic chunks using Docling's `HybridChunker`.
4. Store chunk embeddings and metadata in persistent Chroma.
5. Ask questions from the indexed PDF.
6. Upload an Excel evaluation dataset containing questions and reference answers.
7. Generate RAG answers, retrieve top-k chunks, run RAGAS metrics, and export results to Excel.

RAGAS metrics included:

- Faithfulness
- Context Precision
- Context Recall
- Answer Relevancy

PDF conversion runs Docling explicitly on CUDA with OCR disabled, table structure
extraction enabled, single-item processing batches, and a small inter-stage queue
to limit peak memory usage. Heading hierarchy inference uses PDF bookmarks and
numbering without retaining parsed pages for style analysis. Every embedded chunk
is prefixed with its complete section path to improve retrieval of nested examples.

The output Excel contains:

- question
- reference answer
- generated answer
- retrieved chunks
- retrieved chunk metadata
- faithfulness
- context precision
- context recall
- answer relevancy

## Project Structure

```text
rag_ragas_app/
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   └── rag_pipeline.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── styles.css
├── requirements.txt
├── .env.example
└── README.md
```

## Backend Setup

At project root:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

On Windows PowerShell, create it manually or run:

```powershell
copy .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=rag-ragas-app
CHROMA_DIR=./storage/chroma
UPLOAD_DIR=./storage/uploads
RESULTS_DIR=./storage/results
COLLECTION_NAME=pdf_rag_collection
```

Run backend:

```bash
uvicorn backend.main:app --reload
```

Backend will run at:

```text
http://localhost:8000
```

## Frontend Setup

At project root/frontend:

```bash
npm install
npm run dev
```

Frontend will usually run at:

```text
http://localhost:5173
```

## Excel Evaluation Dataset Format

Your Excel file must contain a question column and a reference answer column.

Required question column:

```text
question
```

Accepted reference answer column names:

```text
reference_answer
reference
ground_truth
ground_truths
answer
```

Example:

| question | reference_answer |
|---|---|
| What is the main objective of the document? | The main objective is ... |
| Which method was used? | The method used was ... |

## API Endpoints

### 1. Ingest PDF

```http
POST /ingest-pdf
```

Form data:

- `file`: PDF file

The pipeline uses Docling's token-aware hybrid chunking and preserves document structure in chunk metadata.

### 2. Ask Question

```http
POST /ask
```

JSON body:

```json
{
  "question": "What is this document about?",
  "top_k": 4
}
```

### 3. Run RAGAS Evaluation

```http
POST /evaluate
```

Form data:

- `file`: Excel file
- `top_k`: optional, default 4

### 4. Download Result Excel

```http
GET /download/{filename}
```

## Notes

- Docling hybrid chunk vectors and metadata are stored persistently under `storage/chroma`.
- Evaluation can take time because RAGAS uses LLM calls for metric scoring.
- For best results, ingest the PDF before running evaluation.
- The system uses OpenAI both for embeddings and answer generation.
