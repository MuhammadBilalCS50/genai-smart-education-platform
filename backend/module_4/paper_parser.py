from __future__ import annotations

import base64
import json
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, Field

from backend.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, RESULTS_DIR


PAPER_CHECKER_DIR = RESULTS_DIR / "paper_checker"
PAPER_CHECKER_DIR.mkdir(parents=True, exist_ok=True)


class StudentAnswer(BaseModel):
    question_number: str = Field(description="Question label exactly as printed, for example 1, 2(a), or Q3")
    question_text: str = Field(min_length=1)
    answer_text: str = Field(default="", description="The student's handwritten answer; empty when unanswered")


class ParsedStudentPaper(BaseModel):
    title: str = "Student paper"
    questions: List[StudentAnswer]


PAPER_STRUCTURE_PROMPT = """Read the attached solved examination-paper PDF directly and return
structured question/answer pairs.

The paper contains printed/typed questions and handwritten student answers. Preserve question
numbers and subparts. Inspect both the PDF text and every page image so handwriting is included.
Correct only obvious recognition or spacing mistakes; do not invent missing words or
improve the student's answer. Associate answer text with the nearest preceding question. Keep an
empty answer_text for unanswered questions. Ignore headers, footers, page numbers, instructions,
and marks unless they are part of a question. Return questions in paper order."""


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _parse_pdf_with_llm(pdf_path: Path, prompt: str, schema: Type[SchemaT]) -> SchemaT:
    """Send a PDF directly to a vision-capable OpenAI model and parse structured output."""
    encoded_pdf = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    response = _openai_client().responses.parse(
        model=OPENAI_CHAT_MODEL,
        input=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": pdf_path.name,
                        "file_data": f"data:application/pdf;base64,{encoded_pdf}",
                    },
                    {
                        "type": "input_text",
                        "text": "Parse the attached PDF according to the instructions.",
                    },
                ],
            },
        ],
        text_format=schema,
    )
    if response.output_parsed is None:
        raise ValueError("The model did not return a structured document extraction.")
    return response.output_parsed


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
    parsed = _parse_pdf_with_llm(pdf_path, PAPER_STRUCTURE_PROMPT, ParsedStudentPaper)
    if not parsed.questions:
        raise ValueError("No numbered questions could be extracted from the student paper.")

    paper_id = uuid.uuid4().hex
    record = {
        "paper_id": paper_id,
        "source_filename": state.get("source_filename") or pdf_path.name,
        "title": parsed.title,
        "questions": [question.model_dump() for question in parsed.questions],
    }
    _save_json("paper", paper_id, record)
    result = {key: record[key] for key in ("paper_id", "source_filename", "title", "questions")}
    return {"paper_id": paper_id, "paper": record, "result": result}


def load_student_paper(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"paper": _load_json("paper", state["paper_id"])}
