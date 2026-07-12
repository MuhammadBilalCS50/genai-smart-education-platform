from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    OPENAI_EMBEDDING_MODEL,
    RESULTS_DIR,
    UPLOAD_DIR,
)

INDEX_SCHEMA = "docling_hybrid_v1"


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def _vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _load_pdf_pages(pdf_path: Path) -> tuple[Any, str]:
    """Parse a PDF with Docling and return its document model and Markdown."""
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
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
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
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
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

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
        text = f"Section: {heading_path}\n\n{contextualized_text}" if heading_path else contextualized_text
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
    chunks_path = RESULTS_DIR / "chunks.txt"
    with chunks_path.open("w", encoding="utf-8") as file:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the configured Chroma collection.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF to ingest")
    args = parser.parse_args()
    print(json.dumps(ingest_pdf(args.pdf_path), indent=2, default=str))


if __name__ == "__main__":
    main()
