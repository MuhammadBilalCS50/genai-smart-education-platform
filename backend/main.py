from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import traceback
from fastapi.responses import JSONResponse

from backend.config import RESULTS_DIR, UPLOAD_DIR
from backend.rag_pipeline import ask_question, clear_chat_history, ingest_pdf, run_ragas_evaluation

app = FastAPI(title="PDF RAG + RAGAS Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    session_id: str = "default"


@app.get("/")
def root():
    return {"message": "PDF RAG + RAGAS Evaluation API is running"}


@app.post("/ingest-pdf")
async def ingest_pdf_endpoint(
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    temp_path = UPLOAD_DIR / file.filename
    try:
        temp_path.write_bytes(await file.read())
        return ingest_pdf(temp_path)
    except Exception as exc:
        error_details = traceback.format_exc()
        print(error_details)

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": error_details,
            },
    )


@app.post("/ask")
def ask_endpoint(payload: AskRequest):
    try:
        return ask_question(payload.question, top_k=payload.top_k, session_id=payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/chat/{session_id}")
def clear_chat_endpoint(session_id: str):
    return clear_chat_history(session_id)


@app.post("/evaluate")
async def evaluate_endpoint(file: UploadFile = File(...), top_k: int = Form(2)):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file.")
    temp_path = UPLOAD_DIR / file.filename
    try:
        temp_path.write_bytes(await file.read())
        return await run_ragas_evaluation(temp_path, top_k=top_k)
    except Exception as e:
        error_details = traceback.format_exc()
        print("\n========== FULL EVALUATION ERROR ==========")
        print(error_details)
        print("==========================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_details,
            },
        )


@app.get("/download/{filename}")
def download_result(filename: str):
    path = (RESULTS_DIR / filename).resolve()
    if not str(path).startswith(str(RESULTS_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=Path(filename).name,
    )
