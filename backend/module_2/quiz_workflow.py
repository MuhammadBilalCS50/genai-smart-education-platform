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
from backend.module_2.quiz_export import build_quiz_outputs
from backend.module_2.quiz_generation import answer_questions, generate_questions

QuizAction = Literal["list_books", "parse_toc", "calibrate", "generate"]


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


def _route(state: QuizState) -> QuizAction:
    return state["action"]


@lru_cache(maxsize=1)
def _workflow() -> Any:
    graph = StateGraph(QuizState)
    graph.add_node("route", lambda state: {})
    graph.add_node("list_books", list_books)
    graph.add_node("parse_toc", parse_toc)
    graph.add_node("load_analysis", load_analysis)
    graph.add_node("calibrate", calibrate_pages)
    graph.add_node("load_calibration", load_calibration)
    graph.add_node("select_chunks", select_quiz_sections)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("answer_questions", answer_questions)
    graph.add_node("build_outputs", build_quiz_outputs)
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
