from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.config import OPENAI_CHAT_MODEL
from backend.module_1.chat import run_chat_workflow


class GeneratedQuestion(BaseModel):
    question: str = Field(min_length=1, description="A short-answer question")
    heading: str = Field(default="")
    pages: str = Field(default="", description="Comma-separated source page numbers")


class QuizDraft(BaseModel):
    title: str = "AI-generated quiz"
    questions: List[GeneratedQuestion]


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


def generate_questions(state: Dict[str, Any]) -> Dict[str, Any]:
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


def answer_questions(state: Dict[str, Any]) -> Dict[str, Any]:
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
