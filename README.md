# GenAI Smart Education Platform

A local React and FastAPI application for working with educational PDFs. It provides four tools:

- **Document assistant** — ingest a PDF into Chroma and ask grounded theoretical, numerical, or programming questions.
- **Quiz generator** — select topics from an indexed book and export question and answer PDFs.
- **Slides generator** — generate, revise, and export a grounded PowerPoint deck, with optional AI-generated slide images.
- **Paper checker** — extract a solved paper and mark scheme, grade matching short-answer questions, review marks, and export a PDF report.

The repository also includes a standalone notebook for evaluating the document assistant with RAGAS.

## Important implementation constraints

- PDF ingestion is explicitly configured for an **NVIDIA CUDA device**. There is no CPU fallback in the current code.
- Ingestion has **OCR disabled**. Use PDFs with an extractable text layer; image-only/scanned books will not be indexed correctly.
- The paper checker is different: it sends the uploaded PDFs directly to a vision-capable OpenAI model, so it can inspect handwriting and page images.
- An OpenAI API key is required. Ingestion, retrieval, generation, paper parsing, and evaluation all make OpenAI API calls.
- On the first chat request, `sentence-transformers` may download the configured reranker model from Hugging Face.
- The frontend API URL is hard-coded as `http://localhost:8000` in `frontend/src/App.jsx`.
- The main chat API searches the entire configured Chroma collection. If several books have been ingested, answers can retrieve passages from any of them. Quiz and slide generation do restrict retrieval to the selected book.

## How it works

### 1. Ingestion and document chat

`backend/module_1/ingest.py` uses Docling to:

1. copy the source PDF into `storage/uploads` with a unique ID;
2. parse the PDF with CUDA, table-structure extraction, bookmark/numbering-based heading inference, and OCR disabled;
3. save the exported Markdown in `storage/results`;
4. create contextualized, token-aware chunks with Docling's `HybridChunker`;
5. prefix chunks with their full heading path; and
6. embed them into persistent Chroma with source, heading, page, and Docling metadata.

The chat workflow redacts common Pakistani PII patterns, classifies each query as `theoretical`, `numerical`, or `programming`, and routes it through LangGraph:

- Theoretical questions use similarity retrieval followed by a cross-encoder reranker and a document-grounded answer prompt.
- Numerical and programming questions retrieve relevant passages and use a specialized solving/code prompt.
- Conversation history is held in backend memory by `session_id` and is lost when the server restarts.

### 2. Quiz generation

The quiz workflow lists the Markdown books created during ingestion, asks the chat model to extract topic-level table-of-contents entries, calibrates printed page numbers to actual PDF page numbers, selects matching Chroma chunks, generates short-answer questions, answers each through the document chat workflow, and creates two A4 PDFs.

The default is 10 questions. To request another amount, include wording such as `Generate 20 questions` in the additional instructions; the parser accepts 1–100 questions. Quiz difficulty can be `easy`, `medium`, or `hard`.

Table-of-contents analyses, page calibrations, and downloadable quiz PDFs are stored only in backend memory. They expire when the backend restarts.

### 3. Slide generation

Slides reuse the quiz book-selection, contents-extraction, calibration, and page-range selection logic. The slides model creates a structured 3–30 slide draft containing titles, bullets, layout recommendations, visual recommendations, and source pages. A draft can be revised repeatedly from user feedback before export through `python-pptx`.

Image generation is opt-in. It uses the configured image model with 1536×1024, low-quality JPEG output. When enabled, the exported deck uses a matching 3:2 canvas and renders each generated image edge-to-edge with no separate PowerPoint text elements. Without generated images, export produces an editable 16:9 text-and-shape deck.

Drafts and exported presentations are held only in backend memory and expire on restart. Image failures are recorded per slide and do not necessarily abort the whole draft.

### 4. Paper checking

The paper checker:

1. sends a solved-paper PDF directly to the OpenAI Responses API and associates printed questions with handwritten answers;
2. parses a mark-scheme PDF into numbered criteria and maximum marks;
3. classifies question types and grades only matching `short question` entries;
4. allows a reviewer to change the proposed marks; and
5. finalizes totals and creates a PDF report.

MCQs, matching questions, and fill-in-the-blank questions are deliberately excluded from grading. A review can be submitted only once. Parsed paper, mark-scheme, check, and report artifacts persist under `storage/results/paper_checker`.

## Repository layout

```text
genai-smart-education-platform/
├── backend/
│   ├── config.py                 # Environment, model, and storage configuration
│   ├── main.py                   # FastAPI application and HTTP endpoints
│   ├── rag_pipeline.py           # Compatibility exports for Module 1
│   ├── module_1/
│   │   ├── ingest.py             # Docling ingestion and Chroma indexing
│   │   └── chat.py               # PII redaction, retrieval, routing, and chat
│   ├── module_2/
│   │   ├── quiz_content.py       # Books, contents parsing, calibration, chunk selection
│   │   ├── quiz_generation.py    # Grounded question and answer generation
│   │   ├── quiz_export.py        # Question/answer PDF rendering
│   │   └── quiz_workflow.py      # Quiz LangGraph workflow
│   ├── module_3/
│   │   ├── slides_generation.py  # Drafting, revision, and optional images
│   │   ├── slides_export.py      # PowerPoint rendering
│   │   └── slides_workflow.py    # Slides LangGraph workflow
│   └── module_4/
│       ├── paper_parser.py       # Student-paper PDF parsing and persistence
│       ├── marks_scheme_parser.py# Marks scheme PDF parsing and persistence
│       ├── paper_checker.py      # Grading, review, and report generation
│       └── checker_workflow.py   # Paper-checker LangGraph workflow
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.jsx               # All four user interfaces
│       ├── main.jsx
│       └── styles.css
├── ragas_evaluation.ipynb        # Standalone RAGAS evaluation
├── requirements.txt
└── README.md
```

The ignored `storage/` directory is created automatically at runtime:

```text
storage/
├── chroma/                       # Persistent vector collection
├── uploads/                      # Uploaded and copied PDFs
└── results/
    ├── <source-id>.md            # Markdown used by quiz/slides book listing
    ├── chunks.txt                # Debug dump; overwritten by each ingestion
    └── paper_checker/            # Persistent checker JSON and reports
```

## Prerequisites

- Python 3.10 or newer
- Node.js and npm compatible with the current Vite release
- An NVIDIA GPU with a working CUDA environment for book ingestion
- An OpenAI API key with access to the configured chat, embedding, and optional image models
- Internet access for OpenAI calls and the initial reranker-model download

Because the frontend dependencies use `latest` and no lockfile is committed, `npm install` resolves the current package releases.

## Setup

Run all commands from the repository root unless noted otherwise.

### 1. Create a Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Create `.env`

Create `.env` in the project root with at least:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

All supported settings and their code defaults are:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_CHAT_MODEL_SLIDES=gpt-5-mini
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RERANKER_MODEL=BAAI/bge-reranker-base

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=genai-smart-education-platform
LANGSMITH_ENDPOINT=

CHROMA_DIR=./storage/chroma
UPLOAD_DIR=./storage/uploads
RESULTS_DIR=./storage/results
COLLECTION_NAME=pdf_rag_collection
```

Paths may be absolute or relative to the process working directory. The application creates the configured storage directories at import time. To disable tracing, set `LANGSMITH_TRACING=false`. `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT` are accepted as fallbacks for their LangSmith equivalents.

### 3. Start the backend

```bash
uvicorn backend.main:app --reload
```

The API is available at `http://localhost:8000`. Interactive OpenAPI documentation is at `http://localhost:8000/docs`.

### 4. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

For a production bundle:

```bash
cd frontend
npm run build
npm run preview
```

## Using the web application

### Document assistant

1. Open **Student Tools → Student RAG Assistant**.
2. Choose a text-based PDF and select **Ingest PDF**. Ingestion can take several minutes.
3. Enter a question and choose the number of chunks to retrieve.
4. Continue asking follow-up questions in the same browser session, or clear the conversation with **New chat**.

Uploading another PDF adds it to the same Chroma collection; it does not clear previously indexed books.

### Quiz generator

1. Ingest the source book first.
2. Open **Student Tools → AI Quiz Generator** and select the indexed book.
3. Analyze its table of contents.
4. Enter the actual PDF viewer page on which the first extracted topic begins. This determines the offset from the printed contents page numbers.
5. Select one or more topics, a difficulty, and optional instructions.
6. Generate and download either the question paper or the paper with answers.

The book must contain a recognizable table of contents with topic-level entries and printed page numbers. Selected calibrated page ranges must overlap page metadata in the indexed chunks.

### Slides generator

1. Ingest the source book first.
2. Open **Instructor Tools → AI Slides Generator**.
3. Select a book, analyze its contents, calibrate pages, and select topics.
4. Choose 3–30 slides, specify the audience and instructions, and optionally enable image generation.
5. Review the draft, submit revision feedback if needed, then export and download the `.pptx` file.

### Paper checker

1. Open **Instructor Tools → AI Paper Checker**.
2. Upload and extract the scanned solved-paper PDF.
3. Upload and extract the corresponding mark-scheme PDF.
4. Run the checker.
5. Review every proposed short-question mark and edit it if necessary.
6. Submit once to finalize totals, then download the report.

AI extraction and grading can be wrong. Treat reviewer approval as required, especially for unclear handwriting or high-stakes assessment.

## API reference

All request and response bodies are JSON unless an endpoint is described as multipart or a download.

### Health and chat

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Health message |
| `POST` | `/ingest-pdf` | Ingest a multipart PDF field named `file` |
| `POST` | `/ask` | Classify and answer a question |
| `DELETE` | `/chat/{session_id}` | Clear in-memory chat history |

Example chat request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain the main method","top_k":4,"session_id":"demo"}'
```

`session_id` defaults to `default`, and `top_k` defaults to 4. The response includes the query type, answer, redaction logs, retrieved contexts, metadata, and references where applicable.

### Quiz

| Method | Path | Body or result |
|---|---|---|
| `GET` | `/quiz/books` | List Markdown books produced by ingestion |
| `POST` | `/quiz/contents` | `{"book_id":"..."}` |
| `POST` | `/quiz/calibrate` | `{"analysis_id":"...","actual_first_page":12}` |
| `POST` | `/quiz/generate` | `calibration_id`, `selected_section_ids`, `difficulty`, `instructions` |
| `GET` | `/quiz/{quiz_id}/pdf?version=questions` | Download question PDF |
| `GET` | `/quiz/{quiz_id}/pdf?version=answers` | Download answer PDF |

Use the IDs returned by each phase in the next request. Section IDs come from `/quiz/contents` or `/quiz/calibrate`.

### Slides

| Method | Path | Body or result |
|---|---|---|
| `GET` | `/slides/books` | List indexed Markdown books |
| `POST` | `/slides/contents` | `{"book_id":"..."}` |
| `POST` | `/slides/calibrate` | `{"analysis_id":"...","actual_first_page":12}` |
| `POST` | `/slides/generate` | `calibration_id`, section IDs, count, audience, instructions, image flag |
| `POST` | `/slides/{draft_id}/feedback` | `{"feedback":"..."}` |
| `POST` | `/slides/{draft_id}/export` | Export the current draft |
| `GET` | `/slides/{presentation_id}/download` | Download `.pptx` |

Example generation body:

```json
{
  "calibration_id": "returned-by-calibrate",
  "selected_section_ids": ["section-1", "section-2"],
  "slide_count": 10,
  "audience": "Undergraduate students",
  "instructions": "Emphasize worked examples",
  "generate_images": false
}
```

### Paper checker

| Method | Path | Body or result |
|---|---|---|
| `POST` | `/paper-checker/paper` | Multipart PDF field named `file`; returns `paper_id` |
| `POST` | `/paper-checker/mark-scheme` | Multipart PDF field named `file`; returns `mark_scheme_id` |
| `POST` | `/paper-checker/check` | `{"paper_id":"...","mark_scheme_id":"..."}` |
| `POST` | `/paper-checker/{check_id}/submit` | Reviewer marks list |
| `GET` | `/paper-checker/{check_id}/report` | Download submitted report |

Submit example:

```json
{
  "marks": [
    {"question_number": "1", "awarded_marks": 3.5},
    {"question_number": "2(a)", "awarded_marks": 2}
  ]
}
```

Omitted questions retain their AI-proposed marks. Submitted marks must be between zero and the question maximum.

## RAGAS evaluation notebook

RAGAS evaluation is not exposed through the API or frontend. It uses the existing Chroma collection and backend configuration directly.

1. Ingest the PDF or PDFs to evaluate.
2. Install and start Jupyter if it is not already available (Jupyter itself is not listed in `requirements.txt`):

   ```bash
   pip install jupyter
   jupyter notebook ragas_evaluation.ipynb
   ```

3. Edit the first configuration cell:

   ```python
   INPUT_DATASET_PATH = Path(r"D:\path\to\evaluation_dataset.xlsx")
   OUTPUT_METRICS_PATH = Path(r"D:\path\to\ragas_metrics.xlsx")
   TOP_K = 4
   ```

4. Run all cells.

The Excel input must include all three logical columns:

- Question: `question` or `questions`
- Reference answer: `reference_answers`, `reference_answer`, `reference`, `ground_truth`, `ground_truths`, or `answer`
- Category: `category`

Example:

| question | reference_answer | category |
|---|---|---|
| What is the main objective? | The objective is ... | theoretical |
| Calculate the example result. | The result is ... | numerical |

Answers and retrieval details are generated for every row, but RAGAS metrics are calculated only when `category`, ignoring case, is `theoretical`. Other categories are written to the output with empty metric cells.

The output workbook contains:

- `results` — input data, generated and redacted questions, generated answers, query classifications, retrieved chunks and metadata, PII logs, and per-row scores;
- `summary` — mean faithfulness, context precision, context recall, and answer relevancy.

The notebook calls the metrics directly and writes the result to `OUTPUT_METRICS_PATH`; the FastAPI server does not need to be running.

## Persistence and lifecycle

| Data | Storage | Survives backend restart? |
|---|---|---|
| Uploaded/copied PDFs | `storage/uploads` | Yes |
| Markdown and Chroma chunks | `storage/results`, `storage/chroma` | Yes |
| Chat history | Process memory | No |
| TOC analyses and calibrations | Process memory | No |
| Quiz PDFs | Process memory | No |
| Slide drafts and `.pptx` exports | Process memory | No |
| Paper-checker extraction/check JSON | `storage/results/paper_checker` | Yes |
| Submitted marks reports | `storage/results/paper_checker` | Yes |

## Troubleshooting

- **CUDA/device error during ingestion:** verify the NVIDIA driver, CUDA-capable PyTorch/Docling environment, and GPU availability. The code currently requests `AcceleratorDevice.CUDA` unconditionally.
- **No extractable text:** ingestion disables OCR. Use a PDF with a text layer or change the pipeline implementation to enable OCR.
- **No books in quiz/slides:** ingest a PDF first and confirm a `<source-id>.md` file exists in `storage/results`.
- **No table-of-contents sections:** the model could not find chapter-child topics with printed page numbers in the Markdown contents excerpt.
- **No indexed chunks in selected ranges:** recheck the actual page entered during calibration and make sure the selected book matches its Chroma source ID.
- **Reranker download/model error:** ensure Hugging Face is reachable or point `RERANKER_MODEL` to an available compatible cross-encoder.
- **Expired ID or missing download:** restart-sensitive quiz/slide state was lost; repeat that workflow after restarting the backend.
- **Frontend cannot reach the API:** start the backend on port 8000 or update the `API` constant in `frontend/src/App.jsx`.
