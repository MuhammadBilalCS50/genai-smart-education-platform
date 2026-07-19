from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.module_4.marks_scheme_parser import load_mark_scheme, parse_mark_scheme
from backend.module_4.paper_checker import check_paper, finalize_check, load_check
from backend.module_4.paper_parser import load_student_paper, parse_student_paper


CheckerAction = Literal["parse_paper", "parse_mark_scheme", "check", "submit"]


class CheckerState(TypedDict, total=False):
    action: CheckerAction
    paper_path: str
    mark_scheme_path: str
    source_filename: str
    paper_id: str
    mark_scheme_id: str
    check_id: str
    marks: List[Dict[str, Any]]
    paper: Dict[str, Any]
    mark_scheme: Dict[str, Any]
    check: Dict[str, Any]
    result: Dict[str, Any]


def _route(state: CheckerState) -> CheckerAction:
    return state["action"]


@lru_cache(maxsize=1)
def _workflow() -> Any:
    graph = StateGraph(CheckerState)
    graph.add_node("route", lambda state: {})
    graph.add_node("parse_paper", parse_student_paper)
    graph.add_node("parse_mark_scheme", parse_mark_scheme)
    graph.add_node("load_paper", load_student_paper)
    graph.add_node("load_mark_scheme", load_mark_scheme)
    graph.add_node("check", check_paper)
    graph.add_node("load_check", load_check)
    graph.add_node("submit", finalize_check)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _route, {
        "parse_paper": "parse_paper",
        "parse_mark_scheme": "parse_mark_scheme",
        "check": "load_paper",
        "submit": "load_check",
    })
    graph.add_edge("parse_paper", END)
    graph.add_edge("parse_mark_scheme", END)
    graph.add_edge("load_paper", "load_mark_scheme")
    graph.add_edge("load_mark_scheme", "check")
    graph.add_edge("check", END)
    graph.add_edge("load_check", "submit")
    graph.add_edge("submit", END)
    return graph.compile()


def run_checker_workflow(action: CheckerAction, **payload: Any) -> Dict[str, Any]:
    """Run one resumable phase of the AI paper-checking LangGraph workflow."""
    state = _workflow().invoke({"action": action, **payload})
    return state["result"]
