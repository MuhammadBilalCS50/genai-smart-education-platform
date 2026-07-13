from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    RERANKER_MODEL,
)

INDEX_SCHEMA = "docling_hybrid_v1"

SYSTEM_PROMPT = """
You are a document-grounded assistant.

Answer the user's question using the supplied context.

Instructions:
- Examine all passages before deciding whether the answer is available.
- Combine evidence across multiple passages where necessary.
- Treat headings, abbreviations, synonyms, and paraphrases as related when justified.
- Make straightforward inferences supported by the context.
- Provide the supported part of an answer even if some details are missing.
- Do not use external knowledge.
- Only state that the document lacks sufficient information when none of the
  supplied passages materially addresses the question.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Conversation history:\n{chat_history}\n\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer:"),
])

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the latest user question as a standalone search query.

Use the conversation history only to resolve references such as:
- it, they, that, those
- the plan, the policy, the method
- the previous section or topic

Rules:
1. Preserve the exact intent of the latest question.
2. Include only context required to understand that question.
3. Do not answer the question.
4. Do not include unrelated earlier topics.
5. If the question is already standalone, return it unchanged.
6. Return only the rewritten query.""",
    ),
    (
        "human",
        """Conversation history:
{chat_history}

Latest question:
{question}

Standalone search query:""",
    ),
])

QUERY_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Classify the user's query using only its wording.

Categories:
- theoretical: asks for an explanation, definition, comparison, description, or conceptual answer.
- numerical: asks to calculate, solve, derive a numeric result, or work through a quantitative problem.
- programming: asks to write, design, explain, debug, or provide code, an algorithm, or pseudocode.

Choose exactly one category. A request to implement a concept is programming even when it also asks for explanation. A request to compute a value is numerical even when it mentions a concept.""",
    ),
    ("human", "Query:\n{question}"),
])

SPECIALIZED_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are solving a {query_type} question.

For a numerical question, show the method, substitutions, calculations, and final result clearly.
For a programming question, provide correct code or pseudocode and a concise explanation.

Use the retrieved document passages as conceptual support. You may perform the requested calculation or construct the requested program yourself. Do not claim that a heading or page supports something unless it appears in the supplied passage metadata. If the retrieved passages are not sufficient, say what is missing while still completing any part that can be solved from the query itself.""",
    ),
    (
        "human",
        """Query:
{question}

Relevant document passages and locations:
{context}

Answer:""",
    ),
])

CHAT_HISTORIES: Dict[str, InMemoryChatMessageHistory] = {}

PII_PATTERNS = {
    "CNIC": r"\b\d{5}-\d{7}-\d{1}\b",
    "PHONE_PK": r"(?<!\w)(?:\+92|0)3\d{2}[\s-]?\d{7}\b",
    "IBAN_PK": r"\bPK\d{2}[A-Z]{4}\d{16}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "NTN": r"\b\d{7}-\d{1}\b",
    "ACCOUNT_NO": r"\b\d{10,20}\b",
}

QueryType = Literal["theoretical", "numerical", "programming"]


class QueryClassification(BaseModel):
    query_type: QueryType = Field(description="The single best category for the query")


class ChatWorkflowState(TypedDict, total=False):
    question: str
    redacted_question: str
    pii_redaction_log: List[str]
    top_k: int
    session_id: str
    use_memory: bool
    query_type: QueryType
    documents: List[Document]
    result: Dict[str, Any]


def redact_pii(text: str) -> tuple[str, List[str]]:
    """Replace Pakistani PII values with placeholder tokens."""
    redacted = text
    log = []
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted)
        if matches:
            log.append(f"Redacted {len(matches)} {label} value(s)")
            redacted = re.sub(pattern, f"[{label}_REDACTED]", redacted)
    return redacted, log


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature)


def _vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


@lru_cache(maxsize=8)
def _reranker(top_n: int) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def _chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[session_id] = InMemoryChatMessageHistory()
    return CHAT_HISTORIES[session_id]


def clear_chat_history(session_id: str) -> Dict[str, Any]:
    CHAT_HISTORIES.pop(session_id, None)
    return {"message": "Chat history cleared", "session_id": session_id}


def _format_chat_history(history: InMemoryChatMessageHistory, max_messages: int = 8) -> str:
    messages = history.messages[-max_messages:]
    if not messages:
        return "No previous conversation."

    formatted = []
    for message in messages:
        role = "User" if message.type == "human" else "Assistant"
        formatted.append(f"{role}: {message.content}")
    return "\n".join(formatted)


def _rerank_documents(docs: List[Document], query: str, top_k: int) -> List[Document]:
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = _reranker(top_k).predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def rewrite_retrieval_query(question: str, history: InMemoryChatMessageHistory) -> str:
    """Convert a conversational question into a standalone retrieval query."""
    if not history.messages:
        return question

    retrieval_history = _format_chat_history(history, max_messages=2)
    chain = QUERY_REWRITE_PROMPT | _llm(temperature=0.0) | StrOutputParser()
    rewritten_query = chain.invoke({
        "chat_history": retrieval_history,
        "question": question,
    }).strip()
    return rewritten_query or question


def ask_question(
    question: str,
    top_k: int = 4,
    session_id: str = "default",
    use_memory: bool = True,
) -> Dict[str, Any]:
    """Retrieve top-k chunks and generate a grounded answer."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    redacted_question, pii_redaction_log = redact_pii(question)
    history = _chat_history(session_id) if use_memory else InMemoryChatMessageHistory()
    chat_history_text = _format_chat_history(history)
    retrieval_query = rewrite_retrieval_query(redacted_question, history)

    candidates = _vectorstore().similarity_search(
        retrieval_query,
        k=max(top_k * 4, 10),
        filter={"index_schema": INDEX_SCHEMA},
    )
    docs = _rerank_documents(candidates, retrieval_query, top_k=top_k)
    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n---\n\n".join(contexts)

    chain = PROMPT | _llm(temperature=0.0)
    answer = chain.invoke({
        "chat_history": chat_history_text,
        "question": redacted_question,
        "context": context_text,
    }).content
    redacted_answer, answer_pii_redaction_log = redact_pii(answer)

    if use_memory:
        history.add_user_message(redacted_question)
        history.add_ai_message(redacted_answer)

    return {
        "question": question,
        "session_id": session_id,
        "redacted_question": redacted_question,
        "chat_history": chat_history_text,
        "pii_redaction_log": pii_redaction_log,
        "answer_pii_redaction_log": answer_pii_redaction_log,
        "answer": redacted_answer,
        "contexts": contexts,
        "metadata": [doc.metadata for doc in docs],
    }


def _classify_query(state: ChatWorkflowState) -> Dict[str, Any]:
    redacted_question, pii_redaction_log = redact_pii(state["question"])
    classifier = _llm(temperature=0.0).with_structured_output(QueryClassification)
    classification = (QUERY_CLASSIFICATION_PROMPT | classifier).invoke({
        "question": redacted_question,
    })
    return {
        "redacted_question": redacted_question,
        "pii_redaction_log": pii_redaction_log,
        "query_type": classification.query_type,
    }


def _route_query(state: ChatWorkflowState) -> QueryType:
    return state["query_type"]


def _answer_theoretical(state: ChatWorkflowState) -> Dict[str, Any]:
    result = ask_question(
        state["question"],
        top_k=state["top_k"],
        session_id=state["session_id"],
        use_memory=state["use_memory"],
    )
    result.update({"query_type": "theoretical", "references": []})
    return {"result": result}


def _retrieve_specialized_context(state: ChatWorkflowState) -> Dict[str, Any]:
    documents = _vectorstore().similarity_search(
        state["redacted_question"],
        k=state["top_k"],
        filter={"index_schema": INDEX_SCHEMA},
    )
    return {"documents": documents}


def _document_references(documents: List[Document]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        heading = str(document.metadata.get("heading_path") or "Heading unavailable")
        pages = str(document.metadata.get("pages") or "Page unavailable")
        key = (heading, pages)
        if key not in seen:
            references.append({"heading": heading, "pages": pages})
            seen.add(key)
    return references


def _format_specialized_context(documents: List[Document]) -> str:
    if not documents:
        return "No relevant indexed passages were retrieved."

    passages = []
    for index, document in enumerate(documents, start=1):
        heading = document.metadata.get("heading_path") or "Heading unavailable"
        pages = document.metadata.get("pages") or "Page unavailable"
        passages.append(
            f"Passage {index}\nHeading: {heading}\nPages: {pages}\nContent:\n{document.page_content}"
        )
    return "\n\n---\n\n".join(passages)


def _answer_specialized(state: ChatWorkflowState) -> Dict[str, Any]:
    documents = state.get("documents", [])
    history = _chat_history(state["session_id"]) if state["use_memory"] else InMemoryChatMessageHistory()
    chat_history_text = _format_chat_history(history)
    chain = SPECIALIZED_ANSWER_PROMPT | _llm(temperature=0.0)
    answer = chain.invoke({
        "query_type": state["query_type"],
        "question": state["redacted_question"],
        "context": _format_specialized_context(documents),
    }).content
    redacted_answer, answer_pii_redaction_log = redact_pii(answer)

    if state["use_memory"]:
        history.add_user_message(state["redacted_question"])
        history.add_ai_message(redacted_answer)

    return {
        "result": {
            "question": state["question"],
            "session_id": state["session_id"],
            "query_type": state["query_type"],
            "redacted_question": state["redacted_question"],
            "chat_history": chat_history_text,
            "pii_redaction_log": state["pii_redaction_log"],
            "answer_pii_redaction_log": answer_pii_redaction_log,
            "answer": redacted_answer,
            "contexts": [document.page_content for document in documents],
            "metadata": [document.metadata for document in documents],
            "references": _document_references(documents),
        },
    }


@lru_cache(maxsize=1)
def _chat_workflow() -> Any:
    workflow = StateGraph(ChatWorkflowState)
    workflow.add_node("classify_query", _classify_query)
    workflow.add_node("answer_theoretical", _answer_theoretical)
    workflow.add_node("retrieve_specialized_context", _retrieve_specialized_context)
    workflow.add_node("answer_specialized", _answer_specialized)

    workflow.add_edge(START, "classify_query")
    workflow.add_conditional_edges(
        "classify_query",
        _route_query,
        {
            "theoretical": "answer_theoretical",
            "numerical": "retrieve_specialized_context",
            "programming": "retrieve_specialized_context",
        },
    )
    workflow.add_edge("answer_theoretical", END)
    workflow.add_edge("retrieve_specialized_context", "answer_specialized")
    workflow.add_edge("answer_specialized", END)
    return workflow.compile()


def run_chat_workflow(
    question: str,
    top_k: int = 4,
    session_id: str = "default",
    use_memory: bool = True,
) -> Dict[str, Any]:
    """Classify a query and run its theoretical, numerical, or programming path."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    final_state = _chat_workflow().invoke({
        "question": question,
        "top_k": top_k,
        "session_id": session_id,
        "use_memory": use_memory,
    })
    return final_state["result"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question using the configured Chroma collection.")
    parser.add_argument("question", help="Question to ask about the indexed PDF")
    parser.add_argument("--top-k", type=int, default=4, help="Number of reranked chunks to use")
    parser.add_argument("--session-id", default="standalone", help="Conversation session identifier")
    parser.add_argument("--no-memory", action="store_true", help="Do not retain conversation memory")
    args = parser.parse_args()
    result = run_chat_workflow(
        args.question,
        top_k=args.top_k,
        session_id=args.session_id,
        use_memory=not args.no_memory,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
