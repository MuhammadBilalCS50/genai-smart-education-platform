from __future__ import annotations

import json
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.config import OPENAI_CHAT_MODEL, RESULTS_DIR


PAPER_CHECKER_DIR = RESULTS_DIR / "paper_checker"
PAPER_CHECKER_DIR.mkdir(parents=True, exist_ok=True)


class StudentAnswer(BaseModel):
    question_number: str = Field(description="Question label exactly as printed, for example 1, 2(a), or Q3")
    question_text: str = Field(min_length=1)
    answer_text: str = Field(default="", description="The student's handwritten answer; empty when unanswered")


class ParsedStudentPaper(BaseModel):
    title: str = "Student paper"
    questions: List[StudentAnswer]


PAPER_STRUCTURE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Turn OCR from a solved examination paper into structured question/answer pairs.

The paper contains printed/typed questions and handwritten student answers. Preserve question
numbers and subparts. Correct only obvious OCR spacing mistakes; do not invent missing words or
improve the student's answer. Associate answer text with the nearest preceding question. Keep an
empty answer_text for unanswered questions. Ignore headers, footers, page numbers, instructions,
and marks unless they are part of a question. Return questions in paper order.""",
    ),
    ("human", "OCR by page:\n\n{ocr_text}"),
])


_OCR_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _ocr_engine() -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install the dependencies in requirements.txt and restart the API."
        ) from exc

    # PaddleOCR 3.x and 2.x use different constructor options.
    try:
        return PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    except (TypeError, ValueError):
        try:
            return PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        except (TypeError, ValueError):
            return PaddleOCR(lang="en")


def _v3_lines(result: Any) -> List[str]:
    lines: List[str] = []
    if isinstance(result, (list, tuple)):
        items = result
    elif hasattr(result, "__next__"):
        items = list(result)
    else:
        items = [result]
    for item in items:
        payload = getattr(item, "json", item)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, dict):
            continue
        data = payload.get("res", payload)
        texts = data.get("rec_texts") or data.get("texts") or []
        lines.extend(str(text).strip() for text in texts if str(text).strip())
    return lines


def _v2_lines(result: Any) -> List[str]:
    lines: List[str] = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not isinstance(page, list):
            continue
        for item in page:
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[1], (list, tuple))
                and item[1]
                and isinstance(item[1][0], str)
            ):
                text = item[1][0].strip()
                if text:
                    lines.append(text)
    return lines


def extract_pdf_ocr(pdf_path: Path) -> List[Dict[str, Any]]:
    """Rasterize a PDF and return PaddleOCR text for each page."""
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "PDF OCR dependencies are missing. Install the dependencies in requirements.txt and restart the API."
        ) from exc

    document = pdfium.PdfDocument(str(pdf_path))
    pages: List[Dict[str, Any]] = []
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            try:
                bitmap = page.render(scale=2.5)
                try:
                    image = np.array(bitmap.to_pil().convert("RGB"), copy=True)
                finally:
                    bitmap.close()
                engine = _ocr_engine()
                with _OCR_LOCK:
                    if hasattr(engine, "predict"):
                        raw = engine.predict(image)
                        lines = _v3_lines(raw)
                    else:
                        raw = engine.ocr(image, cls=True)
                        lines = _v2_lines(raw)
                pages.append({"page": page_index + 1, "lines": lines, "text": "\n".join(lines)})
            finally:
                page.close()
    finally:
        document.close()
    if not any(page["lines"] for page in pages):
        raise ValueError("PaddleOCR could not recognize any text in the student paper.")
    return pages


def _format_pages(pages: List[Dict[str, Any]]) -> str:
    return "\n\n".join(f"--- Page {page['page']} ---\n{page['text']}" for page in pages)


def _save_json(kind: str, identifier: str, data: Dict[str, Any]) -> Path:
    path = PAPER_CHECKER_DIR / f"{kind}-{identifier}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_json(kind: str, identifier: str) -> Dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError(f"Invalid {kind.replace('_', ' ')} identifier.")
    path = PAPER_CHECKER_DIR / f"{kind}-{identifier}.json"
    if not path.is_file():
        raise ValueError(f"{kind.replace('_', ' ').title()} not found or expired.")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_student_paper(state: Dict[str, Any]) -> Dict[str, Any]:
    pdf_path = Path(state["paper_path"])
    pages = extract_pdf_ocr(pdf_path)
    parser = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0).with_structured_output(ParsedStudentPaper)
    parsed = (PAPER_STRUCTURE_PROMPT | parser).invoke({"ocr_text": _format_pages(pages)})
    if not parsed.questions:
        raise ValueError("No numbered questions could be extracted from the student paper.")

    paper_id = uuid.uuid4().hex
    record = {
        "paper_id": paper_id,
        "source_filename": state.get("source_filename") or pdf_path.name,
        "title": parsed.title,
        "questions": [question.model_dump() for question in parsed.questions],
        "ocr_pages": pages,
    }
    _save_json("paper", paper_id, record)
    result = {key: record[key] for key in ("paper_id", "source_filename", "title", "questions")}
    return {"paper_id": paper_id, "paper": record, "result": result}


def load_student_paper(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"paper": _load_json("paper", state["paper_id"])}
