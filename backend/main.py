from io import BytesIO
from pathlib import Path
import re
from typing import List, Literal
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from backend.config import UPLOAD_DIR
from backend.module_1.chat import clear_chat_history, run_chat_workflow
from backend.module_1.ingest import ingest_pdf
from backend.module_2.quiz_export import get_quiz_pdf
from backend.module_2.quiz_workflow import run_quiz_workflow
from backend.module_3.slides_export import get_presentation
from backend.module_3.slides_workflow import run_slides_workflow
from backend.module_4.checker_workflow import run_checker_workflow
from backend.module_4.paper_checker import get_marks_report

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


class SlidesGenerationRequest(BaseModel):
    calibration_id: str
    selected_section_ids: List[str]
    slide_count: int = 10
    audience: str = "Students"
    instructions: str = ""
    generate_images: bool = False


class SlidesFeedbackRequest(BaseModel):
    feedback: str


class PaperCheckRequest(BaseModel):
    paper_id: str
    mark_scheme_id: str


class ReviewedMark(BaseModel):
    question_number: str
    awarded_marks: float


class PaperSubmitRequest(BaseModel):
    marks: List[ReviewedMark]


def _safe_upload_name(filename: str | None, prefix: str) -> str:
    original = Path(filename or f"{prefix}.pdf").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original).stem).strip("-.") or prefix
    return f"{prefix}-{uuid.uuid4().hex}-{stem}.pdf"


async def _save_pdf_upload(file: UploadFile, prefix: str) -> Path:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    path = UPLOAD_DIR / _safe_upload_name(file.filename, prefix)
    path.write_bytes(content)
    return path


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


@app.get("/slides/books")
def slides_books_endpoint():
    return run_slides_workflow("list_books")


@app.post("/slides/contents")
def slides_contents_endpoint(payload: QuizBookRequest):
    try:
        return run_slides_workflow("parse_toc", book_id=payload.book_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not analyze the table of contents: {exc}") from exc


@app.post("/slides/calibrate")
def slides_calibration_endpoint(payload: QuizCalibrationRequest):
    try:
        return run_slides_workflow(
            "calibrate",
            analysis_id=payload.analysis_id,
            actual_first_page=payload.actual_first_page,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/slides/generate")
def slides_generation_endpoint(payload: SlidesGenerationRequest):
    try:
        return run_slides_workflow("generate", **payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate the slide draft: {exc}") from exc


@app.post("/slides/{draft_id}/feedback")
def slides_feedback_endpoint(draft_id: str, payload: SlidesFeedbackRequest):
    try:
        return run_slides_workflow("revise", draft_id=draft_id, feedback=payload.feedback)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not revise the slide draft: {exc}") from exc


@app.post("/slides/{draft_id}/export")
def slides_export_endpoint(draft_id: str):
    try:
        return run_slides_workflow("export", draft_id=draft_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not export the presentation: {exc}") from exc


@app.get("/slides/{presentation_id}/download")
def slides_download_endpoint(presentation_id: str):
    try:
        content, filename = get_presentation(presentation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/paper-checker/paper")
async def paper_checker_paper_endpoint(file: UploadFile = File(...)):
    path = await _save_pdf_upload(file, "student-paper")
    try:
        return await run_in_threadpool(
            run_checker_workflow, "parse_paper",
            paper_path=str(path), source_filename=file.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read the student paper: {exc}") from exc


@app.post("/paper-checker/mark-scheme")
async def paper_checker_mark_scheme_endpoint(file: UploadFile = File(...)):
    path = await _save_pdf_upload(file, "mark-scheme")
    try:
        return await run_in_threadpool(
            run_checker_workflow, "parse_mark_scheme",
            mark_scheme_path=str(path), source_filename=file.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read the mark scheme: {exc}") from exc


@app.post("/paper-checker/check")
def paper_checker_check_endpoint(payload: PaperCheckRequest):
    try:
        return run_checker_workflow("check", **payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not mark the paper: {exc}") from exc


@app.post("/paper-checker/{check_id}/submit")
def paper_checker_submit_endpoint(check_id: str, payload: PaperSubmitRequest):
    try:
        return run_checker_workflow(
            "submit", check_id=check_id,
            marks=[item.model_dump() for item in payload.marks],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not submit the reviewed marks: {exc}") from exc


@app.get("/paper-checker/{check_id}/report")
def paper_checker_report_endpoint(check_id: str):
    try:
        content, filename = get_marks_report(check_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(
        BytesIO(content),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
