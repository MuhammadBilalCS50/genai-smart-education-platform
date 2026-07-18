from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.module_2.quiz_content import (
    calibrate_pages,
    list_books,
    load_analysis,
    load_calibration,
    parse_toc,
    select_quiz_sections,
)
from backend.module_3.slides_export import export_slides
from backend.module_3.slides_generation import generate_slides, load_slide_draft, revise_slides

SlidesAction = Literal["list_books", "parse_toc", "calibrate", "generate", "revise", "export"]


class SlidesState(TypedDict, total=False):
    action: SlidesAction
    book_id: str
    analysis_id: str
    calibration_id: str
    draft_id: str
    actual_first_page: int
    selected_section_ids: List[str]
    slide_count: int
    audience: str
    instructions: str
    feedback: str
    books: List[Dict[str, str]]
    book: Dict[str, str]
    sections: List[Dict[str, Any]]
    adjusted_sections: List[Dict[str, Any]]
    delta: int
    chunks: List[Dict[str, Any]]
    deck: Dict[str, Any]
    revision: int
    result: Dict[str, Any]


def _route(state: SlidesState) -> SlidesAction:
    return state["action"]


@lru_cache(maxsize=1)
def _workflow() -> Any:
    graph = StateGraph(SlidesState)
    graph.add_node("route", lambda state: {})
    graph.add_node("list_books", list_books)
    graph.add_node("parse_toc", parse_toc)
    graph.add_node("load_analysis", load_analysis)
    graph.add_node("calibrate", calibrate_pages)
    graph.add_node("load_calibration", load_calibration)
    graph.add_node("select_chunks", select_quiz_sections)
    graph.add_node("generate_slides", generate_slides)
    graph.add_node("load_draft", load_slide_draft)
    graph.add_node("revise_slides", revise_slides)
    graph.add_node("export_slides", export_slides)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _route, {
        "list_books": "list_books",
        "parse_toc": "parse_toc",
        "calibrate": "load_analysis",
        "generate": "load_calibration",
        "revise": "load_draft",
        "export": "load_draft",
    })
    graph.add_edge("list_books", END)
    graph.add_edge("parse_toc", END)
    graph.add_edge("load_analysis", "calibrate")
    graph.add_edge("calibrate", END)
    graph.add_edge("load_calibration", "select_chunks")
    graph.add_edge("select_chunks", "generate_slides")
    graph.add_edge("generate_slides", END)
    graph.add_conditional_edges(
        "load_draft",
        lambda state: state["action"],
        {"revise": "revise_slides", "export": "export_slides"},
    )
    graph.add_edge("revise_slides", END)
    graph.add_edge("export_slides", END)
    return graph.compile()


def run_slides_workflow(action: SlidesAction, **payload: Any) -> Dict[str, Any]:
    """Run one resumable phase of the LangGraph slides workflow."""
    state = _workflow().invoke({"action": action, **payload})
    return state["result"]
