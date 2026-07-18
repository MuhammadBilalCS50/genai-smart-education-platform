from __future__ import annotations

import io
import re
import uuid
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, XPreformatted

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    RESULTS_DIR,
)
from backend.module_1.chat import run_chat_workflow

HASHED_PREFIX = re.compile(
    r"^(?:(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})|[0-9a-fA-F]{16,64})[_-]"
)
QuizAction = Literal["list_books", "parse_toc", "calibrate", "generate"]


class Section(BaseModel):
    id: str = ""
    title: str = Field(min_length=1)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "Section":
        if self.end_page < self.start_page:
            raise ValueError("section end_page cannot be before start_page")
        return self


class TableOfContents(BaseModel):
    sections: List[Section]


class GeneratedQuestion(BaseModel):
    question: str = Field(min_length=1, description="A short-answer question")
    heading: str = Field(default="")
    pages: str = Field(default="", description="Comma-separated source page numbers")


class QuizDraft(BaseModel):
    title: str = "AI-generated quiz"
    questions: List[GeneratedQuestion]


class QuizState(TypedDict, total=False):
    action: QuizAction
    book_id: str
    analysis_id: str
    calibration_id: str
    actual_first_page: int
    selected_section_ids: List[str]
    difficulty: str
    instructions: str
    books: List[Dict[str, str]]
    book: Dict[str, str]
    sections: List[Dict[str, Any]]
    adjusted_sections: List[Dict[str, Any]]
    delta: int
    chunks: List[Dict[str, Any]]
    questions: List[Dict[str, Any]]
    answered_questions: List[Dict[str, Any]]
    quiz_id: str
    result: Dict[str, Any]


TOC_ANALYSES: Dict[str, Dict[str, Any]] = {}
CALIBRATIONS: Dict[str, Dict[str, Any]] = {}
QUIZ_PDFS: Dict[str, Dict[str, Any]] = {}

TOC_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract the book's table of contents into quiz-worthy sections.

Use the page numbers printed in the table of contents, not PDF viewer page numbers.
Treat each main numbered topic as a section (for example, rows 1, 2, 3),
not a broad part that contains several chapters. For each section return its title,
first printed page, and final printed page. Infer a section's final page as the page
immediately before the next section starts. Exclude front matter, answer keys, indexes,
acknowledgements, and other non-teaching matter. Return sections in book order. Do not
invent a section when the contents are unclear.""",
    ),
    ("human", "Markdown extracted from the book:\n\n{markdown}"),
])

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Create a short-question quiz using only the supplied book chunks.

Rules:
- Every item must be a short-answer question, never multiple choice or true/false.
- Match the requested difficulty.
- Follow the user's additional instructions when they do not conflict with grounding.
- Cover the selected sections fairly and avoid duplicates.
- Keep each question concise and answerable from the chunks.
- For every question, copy the most relevant heading and comma-separated page number(s)
  from chunk metadata into heading and pages.
- Generate exactly the requested number of questions for this batch.""",
    ),
    (
        "human",
        "Difficulty: {difficulty}\nSelected sections: {sections}\nQuestions for this batch: {count}\n"
        "Additional instructions: {instructions}\n\nBook chunks:\n{context}",
    ),
])


def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature)


def _vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )


def _book_name(path: Path) -> str:
    name = HASHED_PREFIX.sub("", path.name)
    if name.lower().endswith(".md"):
        name = name[:-3]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.replace("_", " ").replace("-", " ").strip()


def _book_records() -> List[Dict[str, str]]:
    books = []
    for path in sorted(RESULTS_DIR.glob("*.md"), key=lambda item: item.name.lower()):
        books.append({
            "id": path.stem,
            "name": _book_name(path),
            "markdown_file": path.name,
        })
    return books


def _find_book(book_id: str) -> Dict[str, str]:
    book = next((item for item in _book_records() if item["id"] == book_id), None)
    if not book:
        raise ValueError("The selected book is no longer available.")
    return book


def _number_sections(sections: List[Section]) -> List[Dict[str, Any]]:
    numbered = []
    for index, section in enumerate(sections, start=1):
        item = section.model_dump()
        item["id"] = f"section-{index}"
        numbered.append(item)
    return numbered


def _toc_excerpt(markdown: str, max_chars: int = 120_000) -> str:
    """Keep the contents region of a Markdown book within the model context window."""
    match = re.search(r"(?im)^#{1,4}\s+(?:table\s+of\s+)?contents\s*$", markdown)
    start = match.start() if match else 0
    return markdown[start:start + max_chars]


def _parse_pages(value: Any) -> List[int]:
    """Parse scalar, list, or comma-separated Chroma page metadata."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({page for child in value for page in _parse_pages(child)})
    if isinstance(value, int):
        return [value]
    return sorted({int(match) for match in re.findall(r"\d+", str(value))})


def _list_books(_: QuizState) -> Dict[str, Any]:
    books = _book_records()
    return {"books": books, "result": {"books": books}}


def _parse_toc(state: QuizState) -> Dict[str, Any]:
    book = _find_book(state["book_id"])
    markdown = (RESULTS_DIR / book["markdown_file"]).read_text(encoding="utf-8")
    parser = _llm().with_structured_output(TableOfContents)
    toc = (TOC_PROMPT | parser).invoke({"markdown": _toc_excerpt(markdown)})
    sections = _number_sections(toc.sections)
    if not sections:
        raise ValueError("No table-of-contents sections could be extracted from this book.")
    analysis_id = uuid.uuid4().hex
    TOC_ANALYSES[analysis_id] = {"book": book, "sections": sections}
    result = {"analysis_id": analysis_id, "book": book, "sections": sections}
    return {"analysis_id": analysis_id, "book": book, "sections": sections, "result": result}


def _load_analysis(state: QuizState) -> Dict[str, Any]:
    analysis = TOC_ANALYSES.get(state["analysis_id"])
    if not analysis:
        raise ValueError("This contents analysis expired. Select the book again.")
    return {"book": analysis["book"], "sections": analysis["sections"]}


def _calibrate(state: QuizState) -> Dict[str, Any]:
    sections = state["sections"]
    actual_first_page = int(state["actual_first_page"])
    if actual_first_page < 1:
        raise ValueError("actual_first_page must be at least 1")
    delta = actual_first_page - int(sections[0]["start_page"])
    adjusted = [
        {
            **section,
            "actual_start_page": int(section["start_page"]) + delta,
            "actual_end_page": int(section["end_page"]) + delta,
        }
        for section in sections
    ]
    if any(item["actual_start_page"] < 1 for item in adjusted):
        raise ValueError("The page correction produces an invalid page range.")
    calibration_id = uuid.uuid4().hex
    calibration = {
        "book": state["book"],
        "sections": sections,
        "adjusted_sections": adjusted,
        "delta": delta,
    }
    CALIBRATIONS[calibration_id] = calibration
    result = {"calibration_id": calibration_id, "delta": delta, "sections": adjusted}
    return {"calibration_id": calibration_id, "delta": delta, "adjusted_sections": adjusted, "result": result}


def _load_calibration(state: QuizState) -> Dict[str, Any]:
    calibration = CALIBRATIONS.get(state["calibration_id"])
    if not calibration:
        raise ValueError("This page calibration expired. Select the book again.")
    return calibration


def _select_chunks(state: QuizState) -> Dict[str, Any]:
    selected_ids = set(state.get("selected_section_ids", []))
    selected = [item for item in state["adjusted_sections"] if item["id"] in selected_ids]
    if not selected:
        raise ValueError("Select at least one section.")

    raw = _vectorstore().get(
        where={"source_id": state["book"]["id"]},
        include=["documents", "metadatas"],
    )
    chunks = []
    for content, metadata in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        pages = _parse_pages(metadata.get("pages"))
        if any(
            section["actual_start_page"] <= page <= section["actual_end_page"]
            for section in selected
            for page in pages
        ):
            chunks.append({"content": content, "metadata": metadata, "pages": pages})
    if not chunks:
        raise ValueError("No indexed chunks were found in the selected page ranges.")
    return {"chunks": chunks, "sections": selected}


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    passages = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        passages.append(
            f"Chunk {index}\nHeading: {metadata.get('heading_path') or 'Unknown'}\n"
            f"Pages: {metadata.get('pages') or 'Unknown'}\nContent:\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(passages)


def _chunk_batches(chunks: List[Dict[str, Any]], max_chars: int = 240_000) -> List[List[Dict[str, Any]]]:
    """Batch large section selections while ensuring every matching chunk reaches the LLM."""
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_size = 0
    for chunk in chunks:
        size = len(chunk["content"]) + 200
        if current and current_size + size > max_chars:
            batches.append(current)
            current, current_size = [], 0
        current.append(chunk)
        current_size += size
    if current:
        batches.append(current)
    return batches


def _requested_question_count(instructions: str) -> int:
    match = re.search(
        r"(?i)\b(\d{1,3})\s+(?:short(?:[- ]answer)?\s+)?questions?\b",
        instructions,
    )
    return min(max(int(match.group(1)), 1), 100) if match else 10


def _generate_questions(state: QuizState) -> Dict[str, Any]:
    generator = _llm(temperature=0.3).with_structured_output(QuizDraft)
    instructions = state.get("instructions") or "None"
    requested_count = _requested_question_count(instructions)
    batches = _chunk_batches(state["chunks"])
    per_batch = max(1, (requested_count + len(batches) - 1) // len(batches))
    questions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for batch in batches:
        draft = (QUIZ_PROMPT | generator).invoke({
            "difficulty": state.get("difficulty", "medium"),
            "sections": ", ".join(section["title"] for section in state["sections"]),
            "count": per_batch,
            "instructions": instructions,
            "context": _format_chunks(batch),
        })
        for question in draft.questions:
            key = re.sub(r"\W+", " ", question.question.lower()).strip()
            if key not in seen:
                questions.append(question.model_dump())
                seen.add(key)
    questions = questions[:requested_count]
    if not questions:
        raise ValueError("The model did not generate any questions.")
    return {"questions": questions}


def _references(metadata_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    result = []
    seen = set()
    for metadata in metadata_items:
        heading = str(metadata.get("heading_path") or "Heading unavailable")
        pages = str(metadata.get("pages") or "Page unavailable")
        key = (heading, pages)
        if key not in seen:
            result.append({"heading": heading, "pages": pages})
            seen.add(key)
    return result


def _answer_questions(state: QuizState) -> Dict[str, Any]:
    answered = []
    for index, question in enumerate(state["questions"], start=1):
        response = run_chat_workflow(
            question["question"],
            top_k=4,
            session_id=f"quiz-{state['calibration_id']}-{index}",
            use_memory=False,
            source_id=state["book"]["id"],
        )
        answered.append({
            **question,
            "answer": response["answer"],
            "references": _references(response.get("metadata", [])),
        })
    return {"answered_questions": answered}


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


def _pdf_bytes(state: QuizState, include_answers: bool) -> bytes:
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


def _build_outputs(state: QuizState) -> Dict[str, Any]:
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


def _route(state: QuizState) -> QuizAction:
    return state["action"]


@lru_cache(maxsize=1)
def _workflow() -> Any:
    graph = StateGraph(QuizState)
    graph.add_node("route", lambda state: {})
    graph.add_node("list_books", _list_books)
    graph.add_node("parse_toc", _parse_toc)
    graph.add_node("load_analysis", _load_analysis)
    graph.add_node("calibrate", _calibrate)
    graph.add_node("load_calibration", _load_calibration)
    graph.add_node("select_chunks", _select_chunks)
    graph.add_node("generate_questions", _generate_questions)
    graph.add_node("answer_questions", _answer_questions)
    graph.add_node("build_outputs", _build_outputs)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _route, {
        "list_books": "list_books",
        "parse_toc": "parse_toc",
        "calibrate": "load_analysis",
        "generate": "load_calibration",
    })
    graph.add_edge("list_books", END)
    graph.add_edge("parse_toc", END)
    graph.add_edge("load_analysis", "calibrate")
    graph.add_edge("calibrate", END)
    graph.add_edge("load_calibration", "select_chunks")
    graph.add_edge("select_chunks", "generate_questions")
    graph.add_edge("generate_questions", "answer_questions")
    graph.add_edge("answer_questions", "build_outputs")
    graph.add_edge("build_outputs", END)
    return graph.compile()


def run_quiz_workflow(action: QuizAction, **payload: Any) -> Dict[str, Any]:
    """Run one resumable phase of the LangGraph quiz workflow."""
    state = _workflow().invoke({"action": action, **payload})
    return state["result"]


def get_quiz_pdf(quiz_id: str, version: str) -> tuple[bytes, str]:
    quiz = QUIZ_PDFS.get(quiz_id)
    if not quiz:
        raise KeyError("Quiz not found or expired.")
    if version not in {"questions", "answers"}:
        raise ValueError("version must be 'questions' or 'answers'")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", quiz["book_name"]).strip("-") or "quiz"
    suffix = "question-paper" if version == "questions" else "question-paper-with-answers"
    return quiz[version], f"{safe_name}-{suffix}.pdf"
