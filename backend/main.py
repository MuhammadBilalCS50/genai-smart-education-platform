from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from fastapi.responses import JSONResponse

from backend.config import UPLOAD_DIR
from backend.module_1.chat import clear_chat_history, run_chat_workflow
from backend.module_1.ingest import ingest_pdf

app = FastAPI(title="PDF RAG API")

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
    return {"message": "PDF RAG API is running"}


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
        return run_chat_workflow(payload.question, top_k=payload.top_k, session_id=payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/chat/{session_id}")
def clear_chat_endpoint(session_id: str):
    return clear_chat_history(session_id)
