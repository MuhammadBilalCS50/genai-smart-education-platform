from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    RERANKER_MODEL,
)

INDEX_SCHEMA = "docling_hybrid_v1"

SYSTEM_PROMPT = """You are a careful RAG assistant.
Answer ONLY from the provided context. If the context does not contain the answer,
say that the document does not provide enough information.
Keep the answer concise and factual.
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

CHAT_HISTORIES: Dict[str, InMemoryChatMessageHistory] = {}

PII_PATTERNS = {
    "CNIC": r"\b\d{5}-\d{7}-\d{1}\b",
    "PHONE_PK": r"(?<!\w)(?:\+92|0)3\d{2}[\s-]?\d{7}\b",
    "IBAN_PK": r"\bPK\d{2}[A-Z]{4}\d{16}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "NTN": r"\b\d{7}-\d{1}\b",
    "ACCOUNT_NO": r"\b\d{10,20}\b",
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question using the configured Chroma collection.")
    parser.add_argument("question", help="Question to ask about the indexed PDF")
    parser.add_argument("--top-k", type=int, default=4, help="Number of reranked chunks to use")
    parser.add_argument("--session-id", default="standalone", help="Conversation session identifier")
    parser.add_argument("--no-memory", action="store_true", help="Do not retain conversation memory")
    args = parser.parse_args()
    result = ask_question(
        args.question,
        top_k=args.top_k,
        session_id=args.session_id,
        use_memory=not args.no_memory,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
