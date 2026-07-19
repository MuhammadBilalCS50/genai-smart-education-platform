from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from backend.module_4.paper_parser import _load_json, _parse_pdf_with_llm, _save_json


class MarkSchemeItem(BaseModel):
    question_number: str = Field(description="Question label exactly as written")
    max_marks: float = Field(gt=0, description="Maximum marks available for this question or subpart")
    mark_scheme: str = Field(min_length=1, description="Complete criteria, rubric, acceptable answers, and marking notes")


class ParsedMarkScheme(BaseModel):
    title: str = "Mark scheme"
    items: List[MarkSchemeItem]


MARK_SCHEME_PROMPT = """Read the attached mark-scheme PDF directly and extract a structured
examination mark scheme. Inspect both the PDF text and page images. Return one item per
question or independently marked subpart. Preserve question labels. Include every acceptable
answer, point allocation, rubric, instruction, exception, and examiner note relevant to that item
in mark_scheme. Set max_marks to the stated total; when points are listed separately, add them.
Do not create criteria that are absent from the source. Ignore headers and footers."""


def parse_mark_scheme(state: Dict[str, Any]) -> Dict[str, Any]:
    pdf_path = Path(state["mark_scheme_path"])
    parsed = _parse_pdf_with_llm(pdf_path, MARK_SCHEME_PROMPT, ParsedMarkScheme)
    if not parsed.items:
        raise ValueError("No numbered marking criteria could be extracted from the mark scheme.")

    mark_scheme_id = uuid.uuid4().hex
    record = {
        "mark_scheme_id": mark_scheme_id,
        "source_filename": state.get("source_filename") or pdf_path.name,
        "title": parsed.title,
        "items": [item.model_dump() for item in parsed.items],
    }
    _save_json("mark_scheme", mark_scheme_id, record)
    result = {key: record[key] for key in ("mark_scheme_id", "source_filename", "title", "items")}
    return {"mark_scheme_id": mark_scheme_id, "mark_scheme": record, "result": result}


def load_mark_scheme(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"mark_scheme": _load_json("mark_scheme", state["mark_scheme_id"])}
