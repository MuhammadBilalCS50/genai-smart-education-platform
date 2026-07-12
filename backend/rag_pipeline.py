from __future__ import annotations

import json
import shutil
import sys
import types
import uuid
from pathlib import Path
from typing import Any, Dict, List
from functools import lru_cache

import re
import pandas as pd
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    RERANKER_MODEL,
    RESULTS_DIR,
    UPLOAD_DIR,
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
6. Return only the rewritten query."""
    ),
    (
        "human",
        """Conversation history:
{chat_history}

Latest question:
{question}

Standalone search query:"""
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


def _load_pdf_pages(pdf_path: Path) -> tuple[Any, str]:
    """Parse a PDF with Docling and return its document model and Markdown."""
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA,
        ),
        do_ocr=False,
        do_table_structure=True,
        layout_batch_size=1,
        ocr_batch_size=1,
        table_batch_size=1,
        queue_max_size=2,
        heading_hierarchy_options=HeadingHierarchyOptions(
            enabled=True,
            use_bookmarks=True,
            use_numbering=True,
            use_style=False,
        ),
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )
    docling_document = converter.convert(pdf_path).document
    markdown = docling_document.export_to_markdown().strip()
    if not markdown:
        raise ValueError("No extractable text was found in the PDF.")
    return docling_document, markdown


def _page_numbers(metadata: Dict[str, Any]) -> List[int]:
    """Collect page numbers from Docling's nested chunk provenance metadata."""
    pages: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "page_no" and isinstance(child, int):
                    pages.add(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(metadata)
    return sorted(pages)


def ingest_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Parse a PDF with Docling and index token-aware hybrid chunks in Chroma."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    saved_pdf = UPLOAD_DIR / f"{uuid.uuid4().hex}_{pdf_path.name}"
    shutil.copy2(pdf_path, saved_pdf)

    source_id = saved_pdf.stem
    docling_document, markdown = _load_pdf_pages(saved_pdf)
    markdown_path = RESULTS_DIR / f"{source_id}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    chunker = HybridChunker()
    documents: List[Document] = []
    ids: List[str] = []
    for chunk_index, chunk in enumerate(chunker.chunk(dl_doc=docling_document)):
        contextualized_text = chunker.contextualize(chunk).strip()
        if not contextualized_text:
            continue

        docling_metadata = chunk.meta.export_json_dict()
        headings = [heading.strip() for heading in (chunk.meta.headings or []) if heading.strip()]
        heading_path = " > ".join(headings)
        text = (
            f"Section: {heading_path}\n\n{contextualized_text}"
            if heading_path
            else contextualized_text
        )
        documents.append(Document(
            page_content=text,
            metadata={
                "source_file": saved_pdf.name,
                "source_id": source_id,
                "chunk_index": chunk_index,
                "heading_path": heading_path,
                "pages": ",".join(map(str, _page_numbers(docling_metadata))),
                "docling_metadata": json.dumps(docling_metadata, ensure_ascii=False, default=str),
                "index_schema": INDEX_SCHEMA,
            },
        ))
        ids.append(f"{source_id}:{chunk_index}")

    if not documents:
        raise ValueError("Docling did not produce any indexable chunks from the PDF.")

    _vectorstore().add_documents(documents=documents, ids=ids)
    output_path = RESULTS_DIR / "chunks.txt"

    with output_path.open("w", encoding="utf-8") as file:
        for chunk_id, document in zip(ids, documents):
            file.write(f"Chunk ID: {chunk_id}\n")
            file.write(f"Metadata: {document.metadata}\n")
            file.write(document.page_content)
            file.write("\n\n" + "=" * 80 + "\n\n")

    return {
        "message": "PDF indexed successfully",
        "source_file": saved_pdf.name,
        "chunks_added": len(documents),
        "chunking": "docling_hybrid",
        "markdown_file": markdown_path.name,
        "collection": COLLECTION_NAME,
        "chroma_dir": str(CHROMA_DIR),
    }


def rewrite_retrieval_query(
    question: str,
    history: InMemoryChatMessageHistory,
) -> str:
    """Convert a conversational question into a standalone retrieval query."""

    if not history.messages:
        return question

    # Retrieval usually needs only the latest exchange.
    retrieval_history = _format_chat_history(
        history,
        max_messages=2,
    )

    chain = (
        QUERY_REWRITE_PROMPT
        | _llm(temperature=0.0)
        | StrOutputParser()
    )

    rewritten_query = chain.invoke({
        "chat_history": retrieval_history,
        "question": question,
    }).strip()

    # Defensive fallback for an empty model response.
    return rewritten_query or question


def ask_question(
    question: str,
    top_k: int = 4,
    session_id: str = "default",
    use_memory: bool = True,
) -> Dict[str, Any]:
    """Retrieve top-k chunks and generate a grounded answer."""
    redacted_question, pii_redaction_log = redact_pii(question)
    history = _chat_history(session_id) if use_memory else InMemoryChatMessageHistory()
    chat_history_text = _format_chat_history(history)

    retrieval_query = rewrite_retrieval_query(
        question=redacted_question,
        history=history,
    )
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


def _normalise_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower().strip(): c for c in df.columns}
    question_col = lower_map.get("question") or lower_map.get("questions")
    answer_col = (
        lower_map.get("reference_answer")
        or lower_map.get("reference")
        or lower_map.get("ground_truth")
        or lower_map.get("ground_truths")
        or lower_map.get("answer")
    )
    if not question_col or not answer_col:
        raise ValueError(
            "Excel must contain a question column and a reference answer column. "
            "Accepted reference column names: reference_answer, reference, ground_truth, ground_truths, answer."
        )
    out = df[[question_col, answer_col]].copy()
    out.columns = ["question", "reference_answer"]
    out = out.dropna(subset=["question", "reference_answer"])
    out["question"] = out["question"].astype(str)
    out["reference_answer"] = out["reference_answer"].astype(str)
    return out


def _patch_ragas_optional_vertexai_import() -> None:
    """Prevent old RAGAS imports from failing on removed LangChain VertexAI path.

    Some RAGAS versions import `langchain_community.chat_models.vertexai.ChatVertexAI`
    during module initialization even when you are only using OpenAI. Newer
    `langchain-community` versions no longer expose that module. This harmless
    stub satisfies the unused import without downgrading langchain-community.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    vertexai_stub = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - only used to satisfy RAGAS optional import
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "ChatVertexAI stub was called. This project uses OpenAI for RAGAS evaluation, "
                "so VertexAI should not be instantiated."
            )

    vertexai_stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = vertexai_stub


async def _score_single_sample(sample: Any, metrics: List[Any]) -> Dict[str, float]:
    metric_scores: Dict[str, float] = {}
    for metric in metrics:
        score = await metric.single_turn_ascore(sample)
        metric_name = getattr(metric, "name", metric.__class__.__name__)
        metric_scores[metric_name] = float(score) if score is not None else None
    return metric_scores


async def _run_ragas_metrics_direct(records: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Run RAGAS metrics directly without importing or calling ragas.evaluate()."""
    _patch_ragas_optional_vertexai_import()

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
        try:
            from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecisionMetric
        except ImportError:
            from ragas.metrics import LLMContextPrecisionWithoutReference as ContextPrecisionMetric
    except Exception as exc:
        raise RuntimeError(
            "Could not import direct RAGAS metric classes. Make sure RAGAS is installed. "
            "Run: pip install -U ragas langchain-openai openpyxl pandas. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    evaluator_llm = LangchainLLMWrapper(_llm(temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(_embeddings())

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ContextPrecisionMetric(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    results: List[Dict[str, float]] = []
    for record in records:
        sample = SingleTurnSample(
            user_input=record["question"],
            response=record["generated_answer"],
            reference=record["reference_answer"],
            retrieved_contexts=record["retrieved_chunks_list"],
        )
        results.append(await _score_single_sample(sample, metrics))
    return results


async def run_ragas_evaluation(excel_path: Path, top_k: int = 4) -> Dict[str, Any]:
    """Run RAG over an Excel Q/A dataset and save RAGAS metrics + details to Excel.

    Important: this function intentionally does NOT use `from ragas import evaluate`
    or `ragas.evaluate()`, because those imports can trigger unused VertexAI code
    paths in some RAGAS/LangChain version combinations.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    saved_excel = UPLOAD_DIR / f"{uuid.uuid4().hex}_{excel_path.name}"
    shutil.copy2(excel_path, saved_excel)

    eval_df = _normalise_eval_columns(pd.read_excel(saved_excel))

    detail_rows: List[Dict[str, Any]] = []
    ragas_records: List[Dict[str, Any]] = []

    for _, row in eval_df.iterrows():
        result = ask_question(row["question"], top_k=top_k, session_id=f"eval-{uuid.uuid4().hex}", use_memory=False)
        contexts = result["contexts"]

        detail_row = {
            "question": row["question"],
            "redacted_question": result["redacted_question"],
            "pii_redaction_log": "; ".join(result["pii_redaction_log"]),
            "answer_pii_redaction_log": "; ".join(result["answer_pii_redaction_log"]),
            "reference_answer": row["reference_answer"],
            "generated_answer": result["answer"],
            "retrieved_chunks": "\n\n---CHUNK---\n\n".join(contexts),
            "retrieved_metadata": str(result["metadata"]),
        }
        detail_rows.append(detail_row)

        ragas_records.append({
            "question": result["redacted_question"],
            "reference_answer": row["reference_answer"],
            "generated_answer": result["answer"],
            "retrieved_chunks_list": contexts,
        })

    metric_rows = await _run_ragas_metrics_direct(ragas_records)

    final_df = pd.DataFrame(detail_rows)
    metrics_df = pd.DataFrame(metric_rows)

    # Normalize possible RAGAS metric names across versions.
    rename_map = {
        "faithfulness": "faithfulness",
        "Faithfulness": "faithfulness",
        "llm_context_precision_with_reference": "context_precision",
        "llm_context_precision_without_reference": "context_precision",
        "context_precision": "context_precision",
        "LLMContextPrecisionWithReference": "context_precision",
        "LLMContextPrecisionWithoutReference": "context_precision",
        "llm_context_recall": "context_recall",
        "context_recall": "context_recall",
        "LLMContextRecall": "context_recall",
        "answer_relevancy": "answer_relevancy",
        "response_relevancy": "answer_relevancy",
        "ResponseRelevancy": "answer_relevancy",
    }
    metrics_df = metrics_df.rename(columns={c: rename_map.get(c, c) for c in metrics_df.columns})

    for col in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
        final_df[col] = metrics_df[col] if col in metrics_df.columns else None

    summary_rows = []
    for col in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
        if col in final_df.columns:
            summary_rows.append({"metric": col, "mean_score": float(pd.to_numeric(final_df[col], errors="coerce").mean())})
    summary_df = pd.DataFrame(summary_rows)

    output_path = RESULTS_DIR / f"ragas_results_{uuid.uuid4().hex}.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, index=False, sheet_name="results")
        summary_df.to_excel(writer, index=False, sheet_name="summary")

    return {
        "message": "RAGAS evaluation completed using direct metric classes; ragas.evaluate was not called.",
        "rows_evaluated": len(final_df),
        "summary": summary_rows,
        "result_file": output_path.name,
        "download_url": f"/download/{output_path.name}",
    }
