from __future__ import annotations

import io
import re
import uuid
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.config import OPENAI_CHAT_MODEL
from backend.module_4.paper_parser import PAPER_CHECKER_DIR, _load_json, _save_json


QuestionCategory = Literal["short question", "mcq", "match the column", "fill in the blank"]


class QuestionAssessment(BaseModel):
    question_number: str
    category: QuestionCategory
    awarded_marks: Optional[float] = Field(default=None, ge=0)
    max_marks: Optional[float] = Field(default=None, gt=0)
    reason: str = ""


class AssessmentBatch(BaseModel):
    assessments: List[QuestionAssessment]


CHECK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a careful examination marker. For every student question, classify its
question text as exactly one of: short question, mcq, match the column, fill in the blank.

Only mark short questions. Match them to the supplied marking scheme by question label. Apply the
rubric strictly but allow semantically equivalent wording. Award a number from zero through the
scheme maximum and give a concise, specific reason tied to the student's answer and rubric. For
every other category, return null awarded_marks, null max_marks, and an empty reason. Never grade
non-short questions. Do not compensate for OCR uncertainty by inventing answer content.""",
    ),
    (
        "human",
        "Student questions and answers:\n{paper}\n\nMark scheme:\n{mark_scheme}",
    ),
])


def _question_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", value.lower().replace("question", "q"))
    return key[1:] if key.startswith("q") else key


def check_paper(state: Dict[str, Any]) -> Dict[str, Any]:
    paper = state["paper"]
    mark_scheme = state["mark_scheme"]
    marker = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0).with_structured_output(AssessmentBatch)
    batch = (CHECK_PROMPT | marker).invoke({
        "paper": paper["questions"],
        "mark_scheme": mark_scheme["items"],
    })

    scheme_by_number = {_question_key(item["question_number"]): item for item in mark_scheme["items"]}
    paper_by_number = {_question_key(item["question_number"]): item for item in paper["questions"]}
    short_assessments: List[Dict[str, Any]] = []
    for assessment in batch.assessments:
        if assessment.category != "short question":
            continue
        key = _question_key(assessment.question_number)
        scheme = scheme_by_number.get(key)
        student = paper_by_number.get(key)
        if not scheme or not student:
            continue
        awarded = min(max(float(assessment.awarded_marks or 0), 0), float(scheme["max_marks"]))
        short_assessments.append({
            "question_number": student["question_number"],
            "question_text": student["question_text"],
            "answer_text": student.get("answer_text", ""),
            "mark_scheme": scheme["mark_scheme"],
            "awarded_marks": awarded,
            "max_marks": float(scheme["max_marks"]),
            "reason": assessment.reason.strip() or "No marking reason was returned.",
        })
    if not short_assessments:
        raise ValueError("No short questions with matching mark-scheme entries were found.")

    check_id = uuid.uuid4().hex
    record = {
        "check_id": check_id,
        "paper_id": paper["paper_id"],
        "mark_scheme_id": mark_scheme["mark_scheme_id"],
        "paper_filename": paper["source_filename"],
        "mark_scheme_filename": mark_scheme["source_filename"],
        "status": "draft",
        "assessments": short_assessments,
    }
    _save_json("check", check_id, record)
    return {"check_id": check_id, "check": record, "result": record}


def load_check(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"check": _load_json("check", state["check_id"])}


def _report_bytes(record: Dict[str, Any]) -> bytes:
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#0f172a")))
    styles.add(ParagraphStyle("SmallBody", parent=styles["BodyText"], fontSize=8.5, leading=11))
    document = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="AI Paper Checker Marks Report",
    )
    story: List[Any] = [
        Paragraph("AI Paper Checker — Marks Report", styles["ReportTitle"]),
        Paragraph(f"Student paper: {escape(record['paper_filename'])}", styles["BodyText"]),
        Paragraph(f"Mark scheme: {escape(record['mark_scheme_filename'])}", styles["BodyText"]),
        Spacer(1, 5 * mm),
    ]
    rows: List[List[Any]] = [["Question", "Marks", "AI reason", "Review"]]
    for item in record["assessments"]:
        review = "Changed by reviewer" if item.get("reviewed") else "AI mark accepted"
        rows.append([
            Paragraph(escape(str(item["question_number"])), styles["SmallBody"]),
            Paragraph(f"{item['final_marks']:g} / {item['max_marks']:g}", styles["SmallBody"]),
            Paragraph(escape(item["reason"]), styles["SmallBody"]),
            Paragraph(review, styles["SmallBody"]),
        ])
    table = Table(rows, colWidths=[20 * mm, 24 * mm, 100 * mm, 31 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        table,
        Spacer(1, 7 * mm),
        Paragraph(
            f"Total: <b>{record['total_awarded']:g} / {record['total_possible']:g}</b> "
            f"({record['percentage']:.1f}%)",
            styles["Heading2"],
        ),
    ])
    for item in record["assessments"]:
        review = "Changed by reviewer" if item.get("reviewed") else "AI mark accepted"
        story.extend([
            Spacer(1, 4 * mm),
            Paragraph(f"Question {escape(str(item['question_number']))}", styles["Heading3"]),
            Paragraph(f"<b>Question:</b> {escape(item['question_text'])}", styles["BodyText"]),
            Paragraph(f"<b>Student answer:</b> {escape(item['answer_text'] or 'No answer recognized.')}", styles["BodyText"]),
            Paragraph(f"<b>Mark scheme:</b> {escape(item['mark_scheme'])}", styles["BodyText"]),
            Paragraph(f"<b>AI reason:</b> {escape(item['reason'])}", styles["BodyText"]),
            Paragraph(
                f"<b>AI mark:</b> {item['awarded_marks']:g} / {item['max_marks']:g} &nbsp;&nbsp; "
                f"<b>Final mark:</b> {item['final_marks']:g} / {item['max_marks']:g} ({review})",
                styles["BodyText"],
            ),
        ])
    document.build(story)
    return output.getvalue()


def finalize_check(state: Dict[str, Any]) -> Dict[str, Any]:
    check = state["check"]
    if check.get("status") == "submitted":
        raise ValueError("This marks report has already been submitted.")
    edits = {str(item["question_number"]): float(item["awarded_marks"]) for item in state.get("marks", [])}
    finalized = []
    for item in check["assessments"]:
        final_marks = edits.get(str(item["question_number"]), float(item["awarded_marks"]))
        if final_marks < 0 or final_marks > float(item["max_marks"]):
            raise ValueError(
                f"Marks for question {item['question_number']} must be between 0 and {item['max_marks']:g}."
            )
        finalized.append({
            **item,
            "final_marks": final_marks,
            "reviewed": final_marks != float(item["awarded_marks"]),
        })
    total_awarded = sum(item["final_marks"] for item in finalized)
    total_possible = sum(float(item["max_marks"]) for item in finalized)
    check.update({
        "status": "submitted",
        "assessments": finalized,
        "total_awarded": total_awarded,
        "total_possible": total_possible,
        "percentage": (total_awarded / total_possible * 100) if total_possible else 0,
    })
    report_path = PAPER_CHECKER_DIR / f"report-{check['check_id']}.pdf"
    report_path.write_bytes(_report_bytes(check))
    check["report"] = f"/paper-checker/{check['check_id']}/report"
    _save_json("check", check["check_id"], check)
    return {"check": check, "result": check}


def get_marks_report(check_id: str) -> tuple[bytes, str]:
    check = _load_json("check", check_id)
    if check.get("status") != "submitted":
        raise ValueError("Submit the reviewed marks before downloading the report.")
    path = PAPER_CHECKER_DIR / f"report-{check_id}.pdf"
    if not path.is_file():
        path.write_bytes(_report_bytes(check))
    return path.read_bytes(), f"{Path(check['paper_filename']).stem}-marks-report.pdf"
