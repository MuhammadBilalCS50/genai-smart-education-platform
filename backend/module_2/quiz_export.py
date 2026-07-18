from __future__ import annotations

import io
import re
import uuid
from html import escape
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, XPreformatted

QUIZ_PDFS: Dict[str, Dict[str, Any]] = {}


def _markdown_inline(text: str) -> str:
    """Convert safe, common inline Markdown into ReportLab paragraph markup."""
    formatted = escape(text)
    formatted = re.sub(r"`([^`\n]+)`", r'<font name="Courier">\1</font>', formatted)
    formatted = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", formatted)
    formatted = re.sub(r"__(.+?)__", r"<b>\1</b>", formatted)
    formatted = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", formatted)
    formatted = re.sub(r"\[([^\]]+)]\([^)]+\)", r"<u>\1</u>", formatted)
    return formatted


def _markdown_flowables(markdown: str, styles: Any) -> List[Any]:
    """Render LLM Markdown as headings, paragraphs, lists, quotes, and code blocks."""
    body_style = ParagraphStyle(
        "AnswerMarkdownBody",
        parent=styles["BodyText"],
        leftIndent=8,
        leading=15,
        spaceAfter=6,
    )
    list_style = ParagraphStyle(
        "AnswerMarkdownList",
        parent=body_style,
        leftIndent=24,
        firstLineIndent=0,
        bulletIndent=10,
        spaceAfter=3,
    )
    quote_style = ParagraphStyle(
        "AnswerMarkdownQuote",
        parent=body_style,
        leftIndent=18,
        borderColor=colors.HexColor("#94a3b8"),
        borderWidth=0,
        borderPadding=6,
        textColor=colors.HexColor("#475569"),
    )
    code_style = ParagraphStyle(
        "AnswerMarkdownCode",
        parent=styles["Code"],
        leftIndent=8,
        rightIndent=4,
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        backColor=colors.HexColor("#f1f5f9"),
        borderPadding=7,
        spaceAfter=7,
    )
    heading_styles = {
        level: ParagraphStyle(
            f"AnswerMarkdownHeading{level}",
            parent=styles[f"Heading{min(level, 3)}"],
            leftIndent=8,
            fontSize={1: 15, 2: 13, 3: 11}.get(level, 10),
            leading={1: 18, 2: 16, 3: 14}.get(level, 12),
            spaceBefore=7,
            spaceAfter=4,
        )
        for level in range(1, 7)
    }

    flowables: List[Any] = []
    paragraph_lines: List[str] = []
    code_lines: List[str] = []
    in_code_block = False

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines).strip()
            if text:
                flowables.append(Paragraph(_markdown_inline(text), body_style))
            paragraph_lines.clear()

    def flush_code() -> None:
        if code_lines:
            flowables.append(XPreformatted(escape("\n".join(code_lines)), code_style))
            code_lines.clear()

    for line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if re.match(r"^\s*```", line):
            flush_paragraph()
            if in_code_block:
                flush_code()
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            continue

        heading = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            flowables.append(Paragraph(_markdown_inline(heading.group(2)), heading_styles[level]))
            continue

        unordered = re.match(r"^\s*[-+*]\s+(.+)$", line)
        if unordered:
            flush_paragraph()
            flowables.append(Paragraph(_markdown_inline(unordered.group(1)), list_style, bulletText="-"))
            continue

        ordered = re.match(r"^\s*(\d+)[.)]\s+(.+)$", line)
        if ordered:
            flush_paragraph()
            flowables.append(Paragraph(
                f"<b>{ordered.group(1)}.</b> {_markdown_inline(ordered.group(2))}",
                list_style,
            ))
            continue

        quote = re.match(r"^\s*>\s?(.*)$", line)
        if quote:
            flush_paragraph()
            flowables.append(Paragraph(_markdown_inline(quote.group(1)), quote_style))
            continue

        if re.match(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", line):
            flush_paragraph()
            flowables.append(Spacer(1, 5))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    flush_code()
    return flowables


def _pdf_bytes(state: Dict[str, Any], include_answers: bool) -> bytes:
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI Quiz Generator",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("QuizTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=8)
    small = ParagraphStyle("QuizMeta", parent=styles["Normal"], textColor="#475569", spaceAfter=12)
    answer_label_style = ParagraphStyle(
        "QuizAnswerLabel",
        parent=styles["Heading4"],
        leftIndent=8,
        spaceBefore=2,
        spaceAfter=4,
    )
    story = [
        Paragraph("AI Quiz Generator", title_style),
        Paragraph(f"Book: {escape(state['book']['name'])}", small),
        Paragraph(f"Difficulty: {escape(state.get('difficulty', 'medium').title())}", small),
        Paragraph("Sections: " + escape(", ".join(section["title"] for section in state["sections"])), small),
        Spacer(1, 6),
    ]
    for index, item in enumerate(state["answered_questions"], start=1):
        story.append(Paragraph(f"<b>{index}.</b> {escape(item['question'])}", styles["BodyText"]))
        story.append(Spacer(1, 8))
        if include_answers:
            story.append(Paragraph("<b>Answer</b>", answer_label_style))
            story.extend(_markdown_flowables(str(item["answer"]), styles))
            refs = item.get("references") or []
            if refs:
                source_text = "; ".join(f"{ref['heading']} - page(s) {ref['pages']}" for ref in refs)
                story.append(Paragraph(f"<b>Source:</b> {escape(source_text)}", small))
        else:
            story.append(Spacer(1, 18))
    document.build(story)
    return output.getvalue()


def build_quiz_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    quiz_id = uuid.uuid4().hex
    QUIZ_PDFS[quiz_id] = {
        "questions": _pdf_bytes(state, include_answers=False),
        "answers": _pdf_bytes(state, include_answers=True),
        "book_name": state["book"]["name"],
    }
    result = {
        "quiz_id": quiz_id,
        "book": state["book"],
        "difficulty": state.get("difficulty", "medium"),
        "sections": state["sections"],
        "delta": state["delta"],
        "questions": state["answered_questions"],
        "downloads": {
            "questions": f"/quiz/{quiz_id}/pdf?version=questions",
            "answers": f"/quiz/{quiz_id}/pdf?version=answers",
        },
    }
    return {"quiz_id": quiz_id, "result": result}


def get_quiz_pdf(quiz_id: str, version: str) -> tuple[bytes, str]:
    quiz = QUIZ_PDFS.get(quiz_id)
    if not quiz:
        raise KeyError("Quiz not found or expired.")
    if version not in {"questions", "answers"}:
        raise ValueError("version must be 'questions' or 'answers'")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", quiz["book_name"]).strip("-") or "quiz"
    suffix = "question-paper" if version == "questions" else "question-paper-with-answers"
    return quiz[version], f"{safe_name}-{suffix}.pdf"
