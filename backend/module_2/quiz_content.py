from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, model_validator

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    RESULTS_DIR,
)

HASHED_PREFIX = re.compile(
    r"^(?:(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})|[0-9a-fA-F]{16,64})[_-]"
)


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


TOC_ANALYSES: Dict[str, Dict[str, Any]] = {}
CALIBRATIONS: Dict[str, Dict[str, Any]] = {}

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


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.0)


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
    return [
        {
            "id": path.stem,
            "name": _book_name(path),
            "markdown_file": path.name,
        }
        for path in sorted(RESULTS_DIR.glob("*.md"), key=lambda item: item.name.lower())
    ]


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


def list_books(_: Dict[str, Any]) -> Dict[str, Any]:
    books = _book_records()
    return {"books": books, "result": {"books": books}}


def parse_toc(state: Dict[str, Any]) -> Dict[str, Any]:
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


def load_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    analysis = TOC_ANALYSES.get(state["analysis_id"])
    if not analysis:
        raise ValueError("This contents analysis expired. Select the book again.")
    return {"book": analysis["book"], "sections": analysis["sections"]}


def calibrate_pages(state: Dict[str, Any]) -> Dict[str, Any]:
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


def load_calibration(state: Dict[str, Any]) -> Dict[str, Any]:
    calibration = CALIBRATIONS.get(state["calibration_id"])
    if not calibration:
        raise ValueError("This page calibration expired. Select the book again.")
    return calibration


def select_quiz_sections(state: Dict[str, Any]) -> Dict[str, Any]:
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
