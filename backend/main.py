from io import BytesIO
from typing import List, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from fastapi.responses import JSONResponse, StreamingResponse

from backend.config import UPLOAD_DIR
from backend.module_1.chat import clear_chat_history, run_chat_workflow
from backend.module_1.ingest import ingest_pdf
from backend.module_2.workflow import get_quiz_pdf, run_quiz_workflow

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


class QuizBookRequest(BaseModel):
    book_id: str


class QuizCalibrationRequest(BaseModel):
    analysis_id: str
    actual_first_page: int


class QuizGenerationRequest(BaseModel):
    calibration_id: str
    selected_section_ids: List[str]
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    instructions: str = ""


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


@app.get("/quiz/books")
def quiz_books_endpoint():
    return run_quiz_workflow("list_books")


@app.post("/quiz/contents")
def quiz_contents_endpoint(payload: QuizBookRequest):
    try:
        return run_quiz_workflow("parse_toc", book_id=payload.book_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not analyze the table of contents: {exc}") from exc


@app.post("/quiz/calibrate")
def quiz_calibration_endpoint(payload: QuizCalibrationRequest):
    try:
        return run_quiz_workflow(
            "calibrate",
            analysis_id=payload.analysis_id,
            actual_first_page=payload.actual_first_page,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/quiz/generate")
def quiz_generation_endpoint(payload: QuizGenerationRequest):
    try:
        return run_quiz_workflow(
            "generate",
            calibration_id=payload.calibration_id,
            selected_section_ids=payload.selected_section_ids,
            difficulty=payload.difficulty,
            instructions=payload.instructions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate the quiz: {exc}") from exc


@app.get("/quiz/{quiz_id}/pdf")
def quiz_pdf_endpoint(quiz_id: str, version: Literal["questions", "answers"] = "questions"):
    try:
        content, filename = get_quiz_pdf(quiz_id, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        BytesIO(content),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
